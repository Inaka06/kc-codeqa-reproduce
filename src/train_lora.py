import os
import sys
import json
import argparse
import logging
import yaml
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from common import build_prompt_ids, generate_answers, MAX_NEW_TOKENS
import sacrebleu

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class CodeQADataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=512):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eval_batch_size = eval_batch_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        prompt_ids = build_prompt_ids(self.tokenizer, item["code"], item["question"], self.max_length - MAX_NEW_TOKENS)
        answer_ids = self.tokenizer(item["answer"], add_special_tokens=False)["input_ids"][:MAX_NEW_TOKENS - 1]
        answer_ids = answer_ids + [self.tokenizer.eos_token_id]

        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids  # loss hanya di jawaban
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}

class CodeQAPaddingCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8, return_tensors="pt")

    def __call__(self, batch):
        # Right padding for training
        self.tokenizer.padding_side = "right"
        return self.collator(batch)

def compute_dev_bleu(model, tokenizer, dev_jsonl, max_length=512, max_samples=1000, batch_size=16):
    samples = []
    with open(dev_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    if max_samples:  # subset supaya evaluasi per epoch tidak terlalu lama; set None untuk full dev
        samples = samples[:max_samples]
    preds = generate_answers(model, tokenizer, samples, max_length=max_length, batch_size=batch_size)
    refs = [x["answer"] for x in samples]
    return sacrebleu.corpus_bleu(preds, [refs]).score, preds

class BLEUEvaluationCallback:
    def __init__(self, model, tokenizer, dev_jsonl, output_dir, patience=3, max_length=512, eval_batch_size=16):
        self.model = model
        self.tokenizer = tokenizer
        self.dev_jsonl = dev_jsonl
        self.output_dir = Path(output_dir)
        self.patience = patience
        self.max_length = max_length
        self.best_bleu = -1.0
        self.no_improvement_count = 0

    def on_evaluate(self, args, state, control, **kwargs):
        logger.info("Computing dev set greedy BLEU for checkpoint selection...")
        bleu_score, _ = compute_dev_bleu(self.model, self.tokenizer, self.dev_jsonl, self.max_length, batch_size=self.eval_batch_size)
        logger.info(f"Dev BLEU: {bleu_score:.2f}")

        metrics_path = self.output_dir / "dev_bleu_history.jsonl"
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": state.epoch, "dev_bleu": bleu_score}) + "\n")

        if bleu_score > self.best_bleu:
            self.best_bleu = bleu_score
            self.no_improvement_count = 0
            best_dir = self.output_dir / "best_adapter"
            logger.info(f"New best BLEU! Saving best adapter to {best_dir}")
            self.model.save_pretrained(best_dir)
        else:
            self.no_improvement_count += 1
            logger.info(f"No BLEU improvement for {self.no_improvement_count} eval(s). Patience: {self.patience}")
            if self.no_improvement_count >= self.patience:
                logger.info("Early stopping triggered by BLEU patience limit.")
                control.should_training_stop = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--profile", type=str, default=None, help="Path to profile YAML (e.g. laptop.yaml)")
    parser.add_argument("--sanity", action="store_true", help="Run sanity check with 200 samples and 20 steps")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.profile and os.path.exists(args.profile):
        with open(args.profile, "r", encoding="utf-8") as f:
            profile_config = yaml.safe_load(f)
        # Deep merge
        for k, v in profile_config.items():
            if isinstance(v, dict) and k in config:
                config[k].update(v)
            else:
                config[k] = v

    logger.info(f"Effective Configuration:\n{json.dumps(config, indent=2)}")

    model_name = config["model"]["name_or_path"]
    torch_dtype = getattr(torch, config["model"]["torch_dtype"])
    attn_impl = config["model"]["attn_implementation"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        attn_implementation=attn_impl,
        device_map="auto"
    )

    model.config.use_cache = False
    if config["training"].get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()  # wajib untuk LoRA + gradient checkpointing

    # LoRA config
    lora_conf = config["lora"]
    peft_config = LoraConfig(
        task_type=getattr(TaskType, lora_conf["task_type"]),
        r=lora_conf["r"],
        lora_alpha=lora_conf["lora_alpha"],
        lora_dropout=lora_conf["lora_dropout"],
        target_modules=lora_conf["target_modules"],
        bias=lora_conf["bias"]
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Datasets
    max_len = config["training"]["max_length"]
    stats_path = Path("runs/metrics/dataset_stats.json")
    if stats_path.exists():  # pakai max_length hasil P99 dari prepare_data.py (sama dengan evaluate.py)
        max_len = json.load(open(stats_path, encoding="utf-8")).get("max_length", max_len)
    logger.info(f"Using max_length={max_len}")
    eval_bs = config["evaluation"]["eval_batch_size"]
    train_dataset = CodeQADataset("runs/processed/train.jsonl", tokenizer, max_length=max_len)
    dev_dataset = CodeQADataset("runs/processed/dev.jsonl", tokenizer, max_length=max_len)

    if args.sanity:
        logger.info("Sanity mode enabled: slicing datasets to 200 train / 100 dev samples")
        train_dataset.samples = train_dataset.samples[:200]
        dev_dataset.samples = dev_dataset.samples[:100]
        max_epochs = 1
        max_steps = 20
        eval_steps = 10
        save_steps = 10
    else:
        max_epochs = config["training"]["max_epochs"]
        max_steps = -1
        eval_steps = 0  # epoch-based
        save_steps = 500

    training_args = TrainingArguments(
        output_dir="runs/checkpoints",
        per_device_eval_batch_size=eval_bs,
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        warmup_ratio=config["training"]["warmup_ratio"],
        max_grad_norm=config["training"]["max_grad_norm"],
        num_train_epochs=max_epochs,
        max_steps=max_steps,
        fp16=False,
        bf16=True,
        logging_steps=config["training"]["logging_steps"],
        eval_strategy="epoch" if not args.sanity else "steps",
        eval_steps=eval_steps if args.sanity else None,
        save_strategy="epoch" if not args.sanity else "steps",
        save_steps=save_steps if args.sanity else None,
        save_total_limit=config["training"]["save_total_limit"],
        load_best_model_at_end=False,
        report_to="none",
        seed=config["training"]["seed"]
    )

    collator = CodeQAPaddingCollator(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=collator
    )

    # Add BLEU callback
    from transformers import TrainerCallback
    class BLEUCallbackWrapper(TrainerCallback):
        def __init__(self, cb):
            self.cb = cb
        def on_evaluate(self, args, state, control, **kwargs):
            self.cb.on_evaluate(args, state, control, **kwargs)

    if not args.sanity:
        bleu_cb = BLEUEvaluationCallback(model, tokenizer, "runs/processed/dev.jsonl", "runs", patience=config["evaluation"]["patience"], max_length=max_len, eval_batch_size=eval_bs)
        trainer.add_callback(BLEUCallbackWrapper(bleu_cb))

    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume)

    # Save final adapter
    final_dir = Path("runs/last_adapter")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"Training completed. Last adapter saved to {final_dir}")

if __name__ == "__main__":
    main()