"""VPS experiment: frozen CodeBERT encoder + pretrained BART decoder."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time

ROOT = Path(__file__).resolve().parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import torch
from torch import nn
from transformers import BartForConditionalGeneration, BartTokenizer, RobertaModel, RobertaTokenizer
from transformers import get_linear_schedule_with_warmup
from codebert_training import (MODEL_ID, MODEL_REVISION, capture_state, restore_state,
                                    train_group, load_rows, metrics, sha256, save_torch, write_json)

BART_ID = "facebook/bart-base"
BART_REVISION = "aadd2ab0ae0c8268c7c9693540e9904811f36177"


@dataclass
class Settings:
    profile: str = "vps"
    run_tag: str = "v1"
    train_size: int = 56085
    dev_size: int = 1000
    test_size: int = 7000
    epochs: int = 20
    patience: int = 5
    source_length: int = 400
    target_length: int = 32
    effective_batch: int = 64
    micro_batch: int = 2
    learning_rate: float = 5e-5
    seed: int = 42
    device: str = "auto"
    max_session_seconds: float = 0


class Hybrid(nn.Module):
    def __init__(self, encoder, bart):
        super().__init__()
        self.encoder = encoder
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.decoder = bart.model.decoder
        self.lm_head = bart.lm_head
        self.register_buffer("logits_bias", bart.final_logits_bias.detach().clone())
        self.pad_id = bart.config.pad_token_id
        self.start_id = bart.config.decoder_start_token_id
        self.eos_id = bart.config.eos_token_id
        assert encoder.config.hidden_size == bart.config.d_model
        hidden = bart.config.d_model
        self.bridge = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden))
        nn.init.eye_(self.bridge[1].weight)
        nn.init.zeros_(self.bridge[1].bias)
        # No reference to bart.model.encoder is retained by this module.

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()  # Frozen encoder also keeps dropout disabled.
        return self

    def encode(self, ids, mask):
        with torch.no_grad():
            hidden = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        return self.bridge(hidden.to(self.bridge[0].weight.dtype))

    def logits(self, decoder_ids, memory, source_mask):
        output = self.decoder(input_ids=decoder_ids, encoder_hidden_states=memory,
                              encoder_attention_mask=source_mask, use_cache=False, return_dict=True)
        return self.lm_head(output.last_hidden_state) + self.logits_bias

    def forward(self, source_ids, source_mask, target_ids, target_mask):
        # BART teacher forcing: decoder start token + labels shifted right.
        shifted = torch.full_like(target_ids, self.pad_id)
        shifted[:, 0] = self.start_id
        shifted[:, 1:] = target_ids[:, :-1]
        logits = self.logits(shifted, self.encode(source_ids, source_mask), source_mask)
        # Reuse the generic accumulation loop, whose token count excludes column 0.
        # BART labels include BOS, so count that BOS via one extra mask column below.
        labels = target_ids.masked_fill(~target_mask[:, 1:].bool(), -100)
        total = nn.functional.cross_entropy(logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1),
                                            ignore_index=-100, reduction="sum")
        tokens = target_mask[:, 1:].sum()
        return total / tokens, total, tokens

    @torch.no_grad()
    def generate(self, source_ids, source_mask, max_new_tokens):
        self.eval()
        memory = self.encode(source_ids, source_mask)
        generated = torch.full((len(source_ids), 1), self.start_id, device=source_ids.device, dtype=torch.long)
        finished = torch.zeros(len(source_ids), dtype=torch.bool, device=source_ids.device)
        for _ in range(max_new_tokens):
            token = self.logits(generated, memory, source_mask)[:, -1].argmax(-1)
            token = token.masked_fill(finished, self.pad_id)
            generated = torch.cat((generated, token[:, None]), dim=1)
            finished |= token.eq(self.eos_id)
            if finished.all():
                break
        return generated[:, 1:]


def build_model(device):
    encoder = RobertaModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION, attn_implementation="eager")
    # Load checkpoint on CPU, then keep only pretrained decoder/embeddings/LM head.
    bart, loading = BartForConditionalGeneration.from_pretrained(
        BART_ID, revision=BART_REVISION, attn_implementation="eager", output_loading_info=True)
    assert not any(loading.values()), f"BART pretrained weights did not load completely: {loading}"
    assert bart.lm_head.weight is bart.model.decoder.embed_tokens.weight
    model = Hybrid(encoder, bart)
    del bart
    gc.collect()
    model.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if device.type == "cuda":
        model.encoder.half()  # Frozen weights need no FP32 master copy or optimizer state.
    return model.to(device)


def features(rows, source_tokenizer, target_tokenizer, settings, training):
    ids, masks = [], []
    for row in rows:
        tokens = source_tokenizer.tokenize(row["question"]) + [source_tokenizer.sep_token] + source_tokenizer.tokenize(row["code"])
        tokens = [source_tokenizer.cls_token] + tokens[:settings.source_length - 2] + [source_tokenizer.sep_token]
        item = source_tokenizer.convert_tokens_to_ids(tokens)
        masks.append([1] * len(item) + [0] * (settings.source_length - len(item)))
        ids.append(item + [source_tokenizer.pad_token_id] * (settings.source_length - len(item)))
    source = (torch.tensor(ids), torch.tensor(masks))
    if not training:
        return source  # Gold answer is never tokenized for generation.
    target = target_tokenizer([row["answer"] for row in rows], max_length=settings.target_length,
                              truncation=True, padding="max_length", return_tensors="pt")
    # train_group counts mask[:, 1:]; retain all BART labels, including BOS.
    counting_mask = torch.cat((torch.zeros(len(rows), 1, dtype=torch.long), target.attention_mask), dim=1)
    return (*source, target.input_ids, counting_mask)


def select(rows, size, seed):
    indices = sorted(random.Random(seed).sample(range(len(rows)), min(size, len(rows))))
    return [rows[i] for i in indices]


@torch.no_grad()
def predict(model, tensors, tokenizer, settings, device, log):
    answers = []
    for i in range(len(tensors[0])):
        context = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
        with context:
            ids = model.generate(tensors[0][i:i + 1].to(device), tensors[1][i:i + 1].to(device), settings.target_length)
        answers.append(tokenizer.decode(ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False).strip())
        if i == 0 or (i + 1) % 50 == 0 or i + 1 == len(tensors[0]):
            log(f"Decoded {i + 1}/{len(tensors[0])}")
    return answers


def run(settings=None):
    settings = settings or Settings()
    if settings.profile == "smoke":
        settings = replace(settings, source_length=128, effective_batch=8)
    assert settings.profile in ("smoke", "vps")
    assert re.fullmatch(r"[A-Za-z0-9_-]+", settings.run_tag)
    assert 0 < settings.source_length <= 400 and 2 < settings.target_length <= 64
    assert min(settings.train_size, settings.dev_size, settings.test_size, settings.epochs, settings.effective_batch, settings.patience) > 0
    assert 0 < settings.micro_batch <= settings.effective_batch
    assert settings.max_session_seconds >= 0
    result_root = HERE / "codebert_bart_results"
    result_root.mkdir(exist_ok=True)
    with (result_root / f".{settings.profile}_{settings.run_tag}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("This experiment is already running") from None
        return _run(settings, result_root)


def _run(settings, result_root):
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if settings.device == "auto" else settings.device)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.manual_seed(settings.seed)
    random.seed(settings.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(settings.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    sizes = (8, 4, 4, 1) if settings.profile == "smoke" else (settings.train_size, settings.dev_size, settings.test_size, settings.epochs)
    all_train, hashes = load_rows("train")
    all_dev, dev_hashes = load_rows("dev")
    hashes.update(dev_hashes)
    rows = select(all_train, sizes[0], settings.seed)
    dev_rows = select(all_dev, sizes[1], settings.seed + 1)
    del all_train, all_dev
    import transformers
    config = {k: v for k, v in asdict(settings).items() if k not in ("device", "max_session_seconds")}
    config.update(actual_sizes=sizes, encoder_revision=MODEL_REVISION, decoder_revision=BART_REVISION,
                  frozen_encoder=True, bridge="LayerNorm + identity-initialized Linear", decoding="greedy",
                  precision="fp16" if device.type == "cuda" else "fp32", initial_grad_scale=1024,
                  train_ids=[r["id"] for r in rows],
                  dev_ids=[r["id"] for r in dev_rows], data_hashes=hashes, engine_sha256=sha256(__file__),
                  shared_helper_sha256=sha256(ROOT / "codebert_training.py"),
                  torch_version=torch.__version__, transformers_version=transformers.__version__)
    run_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    out = result_root / f"{settings.profile}_{settings.run_tag}_{run_hash}"
    if not out.exists() and list(result_root.glob(f"{settings.profile}_{settings.run_tag}_*")):
        raise RuntimeError("Configuration changed for this run tag. Restore it to resume or choose a new tag.")
    out.mkdir(exist_ok=True)

    def log(message):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {message}"
        print(line, flush=True)
        with (out / "training.log").open("a") as stream:
            stream.write(line + "\n")

    if (out / "config.json").exists():
        assert json.loads((out / "config.json").read_text()) == json.loads(json.dumps(config))
    write_json(out / "config.json", config)
    log(f"Run: {out}; device={device}; source={settings.source_length}; micro={settings.micro_batch}; effective={settings.effective_batch}")
    source_tokenizer = RobertaTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    target_tokenizer = BartTokenizer.from_pretrained(BART_ID, revision=BART_REVISION)
    train = features(rows, source_tokenizer, target_tokenizer, settings, True)
    dev = features(dev_rows, source_tokenizer, target_tokenizer, settings, False)
    model = build_model(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    log(f"Parameters: total={sum(p.numel() for p in model.parameters()):,}; trainable={sum(p.numel() for p in trainable):,}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.AdamW(trainable, lr=settings.learning_rate, weight_decay=0, foreach=False)
    groups = math.ceil(len(rows) / settings.effective_batch)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(groups * sizes[3] * .1), groups * sizes[3])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=1024)
    cursor = dict(epoch=1, next_group=0, updates=0, skipped=0, best_bleu=-1.0, best_epoch=0,
                  no_improvement=0, history=[], train_seconds=0.0, loss_sum=0.0, token_sum=0)
    if (out / "last.pt").exists():
        saved = torch.load(out / "last.pt", map_location="cpu", weights_only=True)
        assert saved["run_hash"] == run_hash
        cursor = restore_state(saved, model, optimizer, scheduler, scaler)
        del saved
        gc.collect()
        log(f"RESUME confirmed: epoch={cursor['epoch']}; next batch={cursor['next_group']}")
    last_checkpoint_time = time.perf_counter()
    checkpoint_position = (cursor["epoch"], cursor["next_group"])

    def checkpoint():
        nonlocal last_checkpoint_time, checkpoint_position
        payload = capture_state(model, optimizer, scheduler, scaler, cursor)
        payload["run_hash"] = run_hash
        save_torch(out / "last.pt", payload)
        checkpoint_position = (cursor["epoch"], cursor["next_group"])
        last_checkpoint_time = time.perf_counter()

    if not (out / "last.pt").exists():
        checkpoint()
    started, reason = time.perf_counter(), "running"
    try:
        while cursor["epoch"] <= sizes[3] and cursor["no_improvement"] < settings.patience:
            epoch = cursor["epoch"]
            order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(settings.seed + epoch))
            while cursor["next_group"] < groups:
                group = cursor["next_group"]
                indices = order[group * settings.effective_batch:(group + 1) * settings.effective_batch]
                tick = time.perf_counter()
                loss, tokens, updated = train_group(model, optimizer, scheduler, scaler,
                                                     tuple(t[indices] for t in train), settings.micro_batch, device, device.type == "cuda")
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                seconds = time.perf_counter() - tick
                cursor["train_seconds"] += seconds
                cursor["loss_sum"] += loss
                cursor["token_sum"] += tokens
                cursor["updates"] += int(updated)
                cursor["skipped"] += int(not updated)
                cursor["next_group"] += 1
                if group == 0 or (group + 1) % 10 == 0 or group + 1 == groups:
                    log(f"Epoch {epoch}/{sizes[3]}: batch {group + 1}/{groups}; token loss={loss/tokens:.4f}; step={seconds:.2f}s; skipped={cursor['skipped']}")
                if cursor["updates"] == 10 and updated:
                    benchmark = {"estimated_epoch_seconds_excluding_dev_io": cursor["train_seconds"] / 10 * groups,
                                 "peak_tensor_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
                                 "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None}
                    write_json(out / "benchmark.json", benchmark)
                    log(f"Initial benchmark: {benchmark}")
                if time.perf_counter() - last_checkpoint_time >= 300:
                    checkpoint()
                if settings.max_session_seconds and time.perf_counter() - started >= settings.max_session_seconds:
                    checkpoint()
                    reason = "time_budget"
                    break
            if reason == "time_budget":
                break
            if cursor["updates"] == 0:
                raise FloatingPointError("No optimizer update succeeded. Inspect FP16 overflow before continuing.")
            scores = metrics(predict(model, dev, target_tokenizer, settings, device, log), dev_rows)
            cursor["history"].append({"epoch": epoch, "train_token_loss": cursor["loss_sum"] / max(1, cursor["token_sum"]), "dev": scores})
            if scores["BLEU"] > cursor["best_bleu"]:
                cursor.update(best_bleu=scores["BLEU"], best_epoch=epoch, no_improvement=0)
                save_torch(out / "best.pt", {"model": model.state_dict(), "run_hash": run_hash})
            else:
                cursor["no_improvement"] += 1
            log(f"Epoch {epoch}: dev BLEU={scores['BLEU']:.2f}; F1={scores['F1']:.2f}; best={cursor['best_epoch']}")
            cursor.update(epoch=epoch + 1, next_group=0, loss_sum=0.0, token_sum=0)
            write_json(out / "history.json", cursor["history"])
            checkpoint()
            if settings.max_session_seconds and time.perf_counter() - started >= settings.max_session_seconds:
                if cursor["epoch"] <= sizes[3] and cursor["no_improvement"] < settings.patience:
                    reason = "time_budget"
                    break
        if reason == "running":
            reason = "early_stopping" if cursor["no_improvement"] >= settings.patience else "max_epochs"
    except KeyboardInterrupt:
        reason = "interrupted"
        log("Interrupted; last complete checkpoint retained")
    except Exception:
        reason = "error"
        raise
    finally:
        status = {"profile": settings.profile, "stop_reason": reason, "training_complete_for_profile": reason in ("max_epochs", "early_stopping"),
                  "full_paper_training_completed": False, "best_epoch": cursor["best_epoch"], "best_dev_BLEU": cursor["best_bleu"],
                  "checkpoint_epoch_next": checkpoint_position[0], "checkpoint_next_group": checkpoint_position[1],
                  "training_seconds_excluding_dev_io": cursor["train_seconds"], "session_wall_seconds": time.perf_counter() - started,
                  "optimizer_updates": cursor["updates"], "skipped_updates": cursor["skipped"]}
        write_json(out / "status.json", status)
        log(json.dumps(status))
    if not status["training_complete_for_profile"]:
        log("Test remains unopened. Resume with the same settings.")
        return out
    del optimizer, scheduler, scaler, train
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    saved = torch.load(out / "best.pt", weights_only=True, map_location="cpu")
    assert saved["run_hash"] == run_hash
    model.load_state_dict(saved["model"], strict=True)
    del saved
    first = predict(model, tuple(t[:2] for t in dev), target_tokenizer, settings, device, log)
    saved = torch.load(out / "best.pt", weights_only=True, map_location="cpu")
    model.load_state_dict(saved["model"], strict=True)
    del saved
    assert first == predict(model, tuple(t[:2] for t in dev), target_tokenizer, settings, device, log)
    log("Reload + greedy consistency: PASS")
    test_rows, test_hashes = load_rows("test")
    test_rows = select(test_rows, sizes[2], settings.seed + 2)
    test = features(test_rows, source_tokenizer, target_tokenizer, settings, False)
    predictions = predict(model, test, target_tokenizer, settings, device, log)
    result = metrics(predictions, test_rows, extra=True)
    result.update(status)
    result.update(evaluation_scope="full_test" if len(test_rows) == 7000 else "hybrid_subset_only", test_ids=[r["id"] for r in test_rows], test_data_hashes=test_hashes,
                  best_checkpoint_sha256=sha256(out / "best.pt"),
                  peak_tensor_mb=torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
                  peak_reserved_mb=torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None)
    write_json(out / "results.json", result)
    with (out / "test_predictions.jsonl").open("w") as stream:
        for row, prediction in zip(test_rows, predictions):
            stream.write(json.dumps({**row, "prediction": prediction}, ensure_ascii=False) + "\n")
    log(json.dumps(result, indent=2))
    for row, prediction in list(zip(test_rows, predictions))[:5]:
        log(f"Q: {row['question']}\nGold: {row['answer']}\nPred: {prediction}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "vps"), default="vps")
    parser.add_argument("--run-tag", default="v1")
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-session-seconds", type=float, default=0)
    run(Settings(**vars(parser.parse_args())))
