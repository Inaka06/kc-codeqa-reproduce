"""CodeQA Python: official CodeBERT model, portable training and safe resume."""

import argparse
import ast
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import gc
import fcntl
import hashlib
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import random
import re
import time
from types import SimpleNamespace

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import torch
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from transformers import get_linear_schedule_with_warmup

BASE = Path(__file__).resolve().parent
REFERENCE = BASE / "references/CodeQA_CodeBERT"
MODEL_ID = "microsoft/codebert-base"
MODEL_REVISION = "3b0952feddeffad0063f274080e3c23d75e7eb39"
PAPER = {"BLEU": 34.86, "ROUGE_L": 30.28, "METEOR": 12.51, "EM": 4.93, "F1": 31.56}


@dataclass
class Settings:
    profile: str = "paper"
    run_tag: str = "v1"
    micro_batch: int = 4
    effective_batch: int = 64  # Appendix B; official python_script.sh uses 32.
    eval_batch: int = 1
    precision: str = "fp16"
    gradient_checkpointing: bool = True
    device: str = "auto"
    seed: int = 42
    resume: bool = True
    max_session_seconds: float = 0  # 0 = until all epochs finish; safe to resume after interruption.


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = Path(path).with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def save_torch(path, value):
    temporary = Path(path).with_suffix(".pt.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def upstream_data():
    # Reuse the original preprocessing without importing its obsolete training CLI.
    tree = ast.parse((REFERENCE / "run.py").read_text())
    names = {"Example", "InputFeatures", "convert_examples_to_features"}
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    assert {node.name for node in nodes} == names
    namespace = {"logger": logging.getLogger("codeqa_preprocessing")}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "upstream_preprocessing", "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


def load_rows(split):
    columns, hashes = {}, {}
    for extension in ("code", "question", "answer"):
        path = BASE / f"data/python/{split}/{split}.{extension}"
        columns[extension] = path.read_text(encoding="utf-8").splitlines()
        hashes[f"{split}.{extension}"] = sha256(path)
    expected = {"train": 56085, "dev": 7000, "test": 7000}[split]
    assert all(len(values) == expected for values in columns.values()), f"Split alignment failed: {split}"
    rows = [{"id": i, **{key: values[i].strip() for key, values in columns.items()}} for i in range(expected)]
    assert all(row[key] for row in rows for key in columns), f"Empty row: {split}"
    return rows, hashes


def tensor_data(rows, tokenizer, training):
    upstream = upstream_data()
    examples = [upstream.Example(row["id"], row["question"], row["code"], row["answer"] if training else "None") for row in rows]
    args = SimpleNamespace(max_source_length=400, max_target_length=30)
    features = upstream.convert_examples_to_features(examples, tokenizer, args, stage="dev" if training else "test")
    keys = ("source_ids", "source_mask", "target_ids", "target_mask") if training else ("source_ids", "source_mask")
    return tuple(torch.tensor([getattr(item, key) for item in features], dtype=torch.long) for key in keys)


def metrics(predictions, rows, extra=False):
    from collections import Counter
    assert len(predictions) == len(rows) > 0
    metric_root = BASE / "source/codeBERT/code/eval"
    bleu = load_module("codeqa_cb_bleu", metric_root / "bleu/google_bleu.py")
    rouge = load_module("codeqa_cb_rouge", metric_root / "rouge/rouge.py")
    hypotheses = {i: [text] for i, text in enumerate(predictions)}
    references = {i: [row["answer"]] for i, row in enumerate(rows)}
    exact, f1, yn, non = [], [], [], []
    for prediction, row in zip(predictions, rows):
        pred, gold = prediction.lower().split(), row["answer"].lower().split()
        em = pred == gold
        common = sum((Counter(pred) & Counter(gold)).values())
        exact.append(em)
        f1.append(2 * common / (len(pred) + len(gold)) if pred or gold else 1.0)
        (yn if row["answer"].lower() in ("yes", "no") else non).append(em)
    corpus, sentence, _ = bleu.corpus_bleu(hypotheses, references)
    result = {"n": len(rows), "BLEU": float(100 * sentence), "corpus_BLEU_diagnostic": float(100 * corpus),
              "ROUGE_L": float(100 * rouge.Rouge().compute_score(references, hypotheses)[0]),
              "EM": 100 * sum(exact) / len(rows), "F1": 100 * sum(f1) / len(rows),
              "yes_no_correct": sum(yn), "yes_no_n": len(yn), "non_yes_no_correct": sum(non), "non_yes_no_n": len(non)}
    if extra:
        # Separate module object keeps Seq2seq's vendored evaluator unchanged.
        path = metric_root / "meteor/meteor.py"
        text = path.read_text().replace("['java', '-jar', '-Xmx{}'.format(mem), METEOR_JAR,", "['java', '-Xmx{}'.format(mem), '-jar', METEOR_JAR,")
        namespace = {"__file__": str(path)}
        exec(compile(text, str(path), "exec"), namespace)
        meteor = None
        try:
            meteor = namespace["Meteor"]()
            result["METEOR"] = float(100 * meteor.compute_score(references, hypotheses)[0])
        except (OSError, RuntimeError, BrokenPipeError) as error:
            result.update(METEOR=None, METEOR_error=str(error))
        finally:
            if meteor is not None:
                meteor.close()
    return result


def build_model(pretrained=True, tiny=False):
    """Tiny random weights are used only by local regression tests, never smoke/paper."""
    if tiny:
        config = RobertaConfig(vocab_size=31, hidden_size=24, num_hidden_layers=1,
                               num_attention_heads=4, intermediate_size=48, max_position_embeddings=514)
    else:
        config = RobertaConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    config._attn_implementation = "eager"
    encoder = (RobertaModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION, config=config,
                                           attn_implementation="eager") if pretrained and not tiny else RobertaModel(config))
    layer = torch.nn.TransformerDecoderLayer(config.hidden_size, config.num_attention_heads)
    decoder = torch.nn.TransformerDecoder(layer, num_layers=6)
    cls = load_module("codeqa_cb_model", REFERENCE / "model.py").Seq2Seq
    return cls(encoder, decoder, config, beam_size=10, max_length=30, sos_id=0, eos_id=2)


@torch.no_grad()
def predict(model, tensors, tokenizer, batch_size, device, log=print):
    model.eval()
    predictions = []
    # Beam search stays FP32: upstream beam uses a -1e20 finished-beam sentinel.
    for offset in range(0, len(tensors[0]), batch_size):
        source, mask = (tensor[offset:offset + batch_size].to(device) for tensor in tensors[:2])
        beams = model(source_ids=source, source_mask=mask)
        for candidates in beams:
            tokens = candidates[0].cpu().tolist()
            if 0 in tokens:
                tokens = tokens[:tokens.index(0)]
            predictions.append(tokenizer.decode(tokens, clean_up_tokenization_spaces=False))
        if offset == 0 or len(predictions) % 100 == 0 or len(predictions) == len(tensors[0]):
            log(f"Decoded {len(predictions)}/{len(tensors[0])}")
    return predictions


def train_group(model, optimizer, scheduler, scaler, batch, micro_batch, device, use_amp):
    """Normalize by all target tokens, including the last incomplete effective batch."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    tokens = int(batch[3][:, 1:].sum())
    assert tokens > 0
    summed = 0.0
    for offset in range(0, len(batch[0]), micro_batch):
        source, mask, target, target_mask = (tensor[offset:offset + micro_batch].to(device) for tensor in batch)
        context = torch.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()
        with context:
            loss, total, count = model(source_ids=source, source_mask=mask, target_ids=target, target_mask=target_mask)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss; last safe checkpoint is retained")
        scaler.scale(total / tokens).backward()
        summed += float(total.detach())
    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    updated = scaler.get_scale() >= scale_before
    if updated:
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return summed, tokens, updated


def capture_state(model, optimizer, scheduler, scaler, cursor):
    return {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "cursor": cursor, "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_state(saved, model, optimizer, scheduler, scaler):
    model.load_state_dict(saved["model"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    scaler.load_state_dict(saved["scaler"])
    torch.set_rng_state(saved["torch_rng"])
    random.setstate(saved["python_rng"])
    if saved["cuda_rng"] and torch.cuda.is_available():
        assert len(saved["cuda_rng"]) == torch.cuda.device_count(), "Visible GPU count changed; preserve CUDA_VISIBLE_DEVICES"
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    return saved["cursor"]


def run(settings=None):
    settings = settings or Settings()
    assert re.fullmatch(r"[A-Za-z0-9_-]+", settings.run_tag)
    assert settings.profile in ("smoke", "paper")
    lock_dir = BASE / "codebert_paper_results"
    lock_dir.mkdir(exist_ok=True)
    with (lock_dir / f".{settings.profile}_{settings.run_tag}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("This profile/run tag is already running. Stop the other process first.") from None
        return _run(settings)


def _run(settings=None):
    settings = settings or Settings()
    assert settings.profile in ("smoke", "paper")
    assert re.fullmatch(r"[A-Za-z0-9_-]+", settings.run_tag), "Use a simple run tag"
    assert settings.precision in ("fp16", "fp32")
    assert 0 < settings.micro_batch <= settings.effective_batch and settings.eval_batch > 0
    assert settings.max_session_seconds >= 0
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if settings.device == "auto" else settings.device)
    if settings.profile == "paper" and device.type != "cuda":
        raise RuntimeError("Paper profile requires CUDA; use smoke to check CPU execution")
    random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(settings.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    use_amp = device.type == "cuda" and settings.precision == "fp16"
    manifest = json.loads((REFERENCE / "UPSTREAM.json").read_text())
    for name, expected in manifest["vendored_sha256"].items():
        assert sha256(REFERENCE / name) == expected, f"Changed reference source: {name}"
    train_rows, hashes = load_rows("train")
    dev_rows, dev_hashes = load_rows("dev")
    hashes.update(dev_hashes)
    epochs, beam, effective = (1, 2, 4) if settings.profile == "smoke" else (20, 10, settings.effective_batch)
    if settings.profile == "smoke":
        train_rows = train_rows[:8]
        dev_rows = random.Random(settings.seed + 1).sample(dev_rows, 4)
    else:
        dev_rows = random.Random(settings.seed + 1).sample(dev_rows, 1000)
    micro = min(settings.micro_batch, effective)
    import transformers
    config = {key: value for key, value in asdict(settings).items() if key not in ("max_session_seconds", "resume", "device")}
    config.update(epochs=epochs, beam=beam, effective_batch=effective, micro_batch=micro,
                  precision="fp16" if use_amp else "fp32", model_id=MODEL_ID, model_revision=MODEL_REVISION,
                  max_source_length=400, max_target_length=30, learning_rate=5e-5, adam_epsilon=1e-8,
                  weight_decay=0.0, warmup_fraction=0.1, gradient_clipping=None, decoder_layers=6,
                  gradient_normalization="effective_batch_target_tokens", data_hashes=hashes,
                  dev_ids=[row["id"] for row in dev_rows], train_size=len(train_rows), dev_selection_size=len(dev_rows),
                  source_manifest=manifest, engine_sha256=sha256(__file__), torch_version=torch.__version__,
                  transformers_version=transformers.__version__, device_type=device.type,
                  evaluator_sha256={str(path.relative_to(BASE)): sha256(path) for path in sorted((BASE / "source/codeBERT/code/eval").rglob("*")) if path.is_file() and "__pycache__" not in path.parts})
    run_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    out = BASE / "codebert_paper_results" / f"{settings.profile}_{settings.run_tag}_{run_hash}"
    siblings = list(out.parent.glob(f"{settings.profile}_{settings.run_tag}_*")) if out.parent.exists() else []
    if settings.resume and not out.exists() and siblings:
        raise RuntimeError("Config/source changed for this run tag. Restore it to resume, or choose a new run_tag.")
    out.mkdir(parents=True, exist_ok=True)
    log_file = out / "training.log"

    def log(message):
        text = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {message}"
        print(text, flush=True)
        with log_file.open("a") as stream:
            stream.write(text + "\n")

    log(f"Run: {out}")
    log(f"Device={device}; precision={config['precision']}; micro={micro}; effective={effective}; epochs={epochs}; beam={beam}")
    if device.type == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(device)}; free/total bytes: {torch.cuda.mem_get_info(device)}")
        torch.cuda.reset_peak_memory_stats(device)
    if (out / "config.json").exists():
        assert json.loads((out / "config.json").read_text()) == config, "Run config mismatch"
    write_json(out / "config.json", config)
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    log("Tokenizing train/dev using official preprocessing")
    train = tensor_data(train_rows, tokenizer, True)
    dev = tensor_data(dev_rows, tokenizer, False)
    # Answers must never be passed to the generative path.
    assert len(dev) == 2 and train[0].shape[1] == 400 and train[2].shape[1] == 30
    exists = (out / "last.pt").exists()
    if exists and not settings.resume:
        raise RuntimeError("Run already exists. Enable resume or choose a new run_tag.")
    model = build_model(pretrained=not exists).to(device)
    model.beam_size = beam
    if settings.gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, eps=1e-8, weight_decay=0.0, foreach=False)
    groups_per_epoch = math.ceil(len(train_rows) / effective)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(groups_per_epoch * epochs * .1), groups_per_epoch * epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    cursor = dict(epoch=1, next_group=0, best_bleu=-1.0, best_epoch=0, history=[],
                  updates=0, skipped_updates=0, train_seconds=0.0, epoch_loss_sum=0.0, epoch_tokens=0)
    if exists:
        saved = torch.load(out / "last.pt", map_location="cpu", weights_only=True)
        assert saved["run_hash"] == run_hash
        cursor = restore_state(saved, model, optimizer, scheduler, scaler)
        del saved
        gc.collect()
        log(f"RESUME confirmed: epoch={cursor['epoch']}; next effective batch={cursor['next_group']}; best epoch={cursor['best_epoch']}")
    log(f"Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    checkpoint_position = (cursor["epoch"], cursor["next_group"])
    last_checkpoint_time = time.perf_counter()

    def checkpoint():
        nonlocal checkpoint_position, last_checkpoint_time
        payload = capture_state(model, optimizer, scheduler, scaler, cursor)
        payload["run_hash"] = run_hash
        save_torch(out / "last.pt", payload)
        checkpoint_position = (cursor["epoch"], cursor["next_group"])
        last_checkpoint_time = time.perf_counter()

    if not exists:
        checkpoint()
    started, reason = time.perf_counter(), "running"
    try:
        while cursor["epoch"] <= epochs:
            epoch = cursor["epoch"]
            order = torch.randperm(len(train_rows), generator=torch.Generator().manual_seed(settings.seed + epoch))
            while cursor["next_group"] < groups_per_epoch:
                group = cursor["next_group"]
                indices = order[group * effective:(group + 1) * effective]
                batch = tuple(tensor[indices] for tensor in train)
                tick = time.perf_counter()
                loss, tokens, updated = train_group(model, optimizer, scheduler, scaler, batch, micro, device, use_amp)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                seconds = time.perf_counter() - tick
                cursor["train_seconds"] += seconds
                cursor["epoch_loss_sum"] += loss
                cursor["epoch_tokens"] += tokens
                cursor["updates"] += int(updated)
                cursor["skipped_updates"] += int(not updated)
                cursor["next_group"] += 1
                if group == 0 or (group + 1) % 10 == 0 or group + 1 == groups_per_epoch:
                    log(f"Epoch {epoch}/{epochs}: effective batch {group + 1}/{groups_per_epoch}; token loss={loss/tokens:.4f}; step={seconds:.2f}s; skipped={cursor['skipped_updates']}")
                if cursor["updates"] == 10 and updated:
                    estimate = cursor["train_seconds"] / 10 * groups_per_epoch
                    benchmark = {"training_seconds_first_10_updates": cursor["train_seconds"],
                                 "estimated_epoch_seconds_excluding_dev_io": estimate,
                                 "peak_tensor_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None}
                    write_json(out / "benchmark.json", benchmark)
                    log(f"Initial benchmark (exclude dev/IO): {benchmark}")
                # Multi-GB optimizer checkpoints are expensive on the VPS home HDD.
                if time.perf_counter() - last_checkpoint_time >= 300:
                    checkpoint()
                if settings.max_session_seconds and time.perf_counter() - started >= settings.max_session_seconds:
                    checkpoint()
                    reason = "time_budget"
                    break
            if reason == "time_budget":
                break
            log(f"Epoch {epoch}: dev beam {beam}, {len(dev_rows)} examples")
            dev_started = time.perf_counter()
            predictions = predict(model, dev, tokenizer, settings.eval_batch, device, log)
            scores = metrics(predictions, dev_rows)
            record = {"epoch": epoch, "train_token_loss": cursor["epoch_loss_sum"] / max(1, cursor["epoch_tokens"]),
                      "dev_seconds": time.perf_counter() - dev_started, "dev": scores}
            cursor["history"].append(record)
            if scores["BLEU"] > cursor["best_bleu"]:
                cursor.update(best_bleu=scores["BLEU"], best_epoch=epoch)
                save_torch(out / "best.pt", {"run_hash": run_hash, "epoch": epoch, "model": model.state_dict()})
            log(f"Epoch {epoch}: dev BLEU={scores['BLEU']:.2f}; EM={scores['EM']:.2f}; F1={scores['F1']:.2f}; best={cursor['best_epoch']}")
            cursor.update(epoch=epoch + 1, next_group=0, epoch_loss_sum=0.0, epoch_tokens=0)
            write_json(out / "history.json", cursor["history"])
            checkpoint()
            if settings.max_session_seconds and time.perf_counter() - started >= settings.max_session_seconds:
                reason = "time_budget" if cursor["epoch"] <= epochs else "max_epochs"
                break
        if reason == "running":
            reason = "max_epochs"
    except KeyboardInterrupt:
        reason = "interrupted"
        log("Interrupted. Last atomic checkpoint retained; an unfinished accumulation group is never saved.")
    except Exception:
        reason = "error"
        raise
    finally:
        status = {"profile": settings.profile, "stop_reason": reason,
                  "training_complete_for_profile": reason == "max_epochs",
                  "full_paper_training_completed": settings.profile == "paper" and reason == "max_epochs",
                  "checkpoint_epoch_next": checkpoint_position[0], "checkpoint_next_effective_batch": checkpoint_position[1],
                  "best_epoch": cursor["best_epoch"], "best_dev_BLEU": cursor["best_bleu"],
                  "training_seconds_total_excluding_dev_io": cursor["train_seconds"],
                  "session_wall_seconds": time.perf_counter() - started}
        write_json(out / "status.json", status)
        log(json.dumps(status))
    if reason != "max_epochs":
        log("Test remains unopened. Rerun with the same settings to resume.")
        return out
    # Free optimizer state before FP32 beam evaluation and checkpoint reload.
    del optimizer, scheduler, scaler, train
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    best = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    assert best["run_hash"] == run_hash
    model.load_state_dict(best["model"], strict=True)
    del best
    probe = predict(model, tuple(t[:2] for t in dev), tokenizer, 1, device, log)
    best = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(best["model"], strict=True)
    del best
    assert probe == predict(model, tuple(t[:2] for t in dev), tokenizer, 1, device, log), "Checkpoint reload changed beam output"
    log("Reload + beam consistency: PASS")
    test_rows, test_hashes = load_rows("test")
    if settings.profile == "smoke":
        test_rows = test_rows[:4]
    test = tensor_data(test_rows, tokenizer, False)
    log(f"Final test: {len(test_rows)} examples; beam {beam}")
    tick = time.perf_counter()
    predictions = predict(model, test, tokenizer, settings.eval_batch, device, log)
    result = metrics(predictions, test_rows, extra=True)
    result.update(status)
    result.update(test_inference_seconds=time.perf_counter() - tick, test_data_hashes=test_hashes,
                  evaluation_scope="full_test" if settings.profile == "paper" else "smoke_subset_only",
                  best_checkpoint_sha256=sha256(out / "best.pt"), paper_python_codebert_reference=PAPER)
    with (out / "test_predictions.jsonl.tmp").open("w") as stream:
        for row, prediction in zip(test_rows, predictions):
            stream.write(json.dumps({**row, "prediction": prediction}, ensure_ascii=False) + "\n")
    (out / "test_predictions.jsonl.tmp").replace(out / "test_predictions.jsonl")
    write_json(out / "results.json", result)
    log(json.dumps(result, indent=2))
    for row, prediction in list(zip(test_rows, predictions))[:5]:
        log(f"Q: {row['question']}\nGold: {row['answer']}\nPred: {prediction}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "paper"), default="paper")
    parser.add_argument("--run-tag", default="v1")
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--effective-batch", type=int, choices=(32, 64), default=64)
    parser.add_argument("--eval-batch", type=int, default=1)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--max-session-seconds", type=float, default=0)
    args = vars(parser.parse_args())
    args["gradient_checkpointing"] = not args.pop("no_gradient_checkpointing")
    run(Settings(**args))
