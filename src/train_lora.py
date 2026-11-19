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
from peft import LoraConfig, get_peft_model, TaskType, print_trainable_parameters
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        code = item["code"]
        question = item["question"]
        answer = item["answer"]

        prompt_prefix = f"### Code\n{code}\n### Question\n{question}\n### Answer\n"
        full_text = prompt_prefix + answer + self.tokenizer.eos_token

        encoded_prompt = self.tokenizer(prompt_prefix, add_special_tokens=False, truncation=True, max_length=self.max_length)
        encoded_full = self.tokenizer(full_text, add_special_tokens=True, truncation=True, max_length=self.max_length)

        input_ids = encoded_full["input_ids"]
        attention_mask = encoded_full["attention_mask"]
        labels = list(input_ids)

        prompt_len = len(encoded_prompt["input_ids"])
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

class CodeQAPaddingCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8, return_tensors="pt")

    def __call__(self, batch):
        # Right padding for training
        self.tokenizer.padding_side = "right"
        return self.collator(batch)

def compute_dev_bleu(model, tokenizer, dev_jsonl, max_new_tokens=32, batch_size=16):
    tokenizer.padding_side = "left"
    model.eval()

    samples = []
    with open(dev_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))

    preds = []
    refs = []

    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        prompts = [f"### Code\n{item['code']}\n### Question\n{item['question']}\n### Answer\n" for item in batch]
        gold = [item["answer"] for item in batch]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decode only generated part
        for j, out in enumerate(outputs):
            prompt_len = inputs["input_ids"][j].shape[0]
            gen_tokens = out[prompt_len:]
            pred = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            preds.append(pred)
            refs.append(gold[j])

    bleu = sacrebleu.corpus_bleu(preds, [refs])
    return bleu.score, preds

class BLEUEvaluationCallback:
    def __init__(self, model, tokenizer, dev_jsonl, output_dir, patience=3):
        self.model = model
        self.tokenizer = tokenizer
        self.dev_jsonl = dev_jsonl
        self.output_dir = Path(output_dir)
        self.patience = patience
        self.best_bleu = -1.0
        self.no_improvement_count = 0

    def on_evaluate(self, args, state, control, **kwargs):
        logger.info("Computing dev set greedy BLEU for checkpoint selection...")
        bleu_score, _ = compute_dev_bleu(self.model, self.tokenizer, self.dev_jsonl)
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

    if config["training"].get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

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
    print_trainable_parameters()

    # Datasets
    max_len = config["training"]["max_length"]
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
        bleu_cb = BLEUEvaluationCallback(model, tokenizer, "runs/processed/dev.jsonl", "runs", patience=config["evaluation"]["patience"])
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
