"""Shared helpers for the CodeQA context-ablation study (E1-E8).

Data loading with hash checks, input templates, per-example metrics through the
official evaluator, and W&B logging helpers. Every scoring path goes through
codeqa_metrics ("codeqa-standard-v1").
"""
from collections import Counter
import hashlib
import json
import os
import random
import sys
import uuid
from pathlib import Path

REPO = Path(os.environ.get("KC_REPO", Path.home() / "inahuy/kacuy/kc-codeqa-reproduce"))
EXP = Path(os.environ.get("KC_EXP", Path.home() / "inahuy/kacuy/codeqa_exp"))
sys.path.insert(0, str(REPO))
import codeqa_metrics  # noqa: E402

DATA = REPO / "data/python"
SPLIT_SIZES = {"train": 56085, "dev": 7000, "test": 7000}
# Hashes recorded in the historical run configs (config.json data_hashes / results.json test_data_hashes).
EXPECTED_SHA256 = {
    "train.code": "0fc4b662c1fbb5738dd3dd1f8aba31c21fe5c0b1add3b2d3f9b0729a28e70185",
    "train.question": "f70382ef035938d63e463fccd2db27d8f9d7555d3ecf519b17f2cd83bed1fdcc",
    "train.answer": "0d66995f838b05a9688b243f9834bcaf5061b4f02a462958a2fea2a3ba78de00",
    "dev.code": "98f2de17772ad34fb4ba730799b6508fbf254be50f98d62bd2244452e5c9c43c",
    "dev.question": "0f231a6794cd214d9ca19ba16e4e2dade483266f5675e43ecaac9a4d874add68",
    "dev.answer": "0a702fecb4a3e4b8d36e04bedee638d32b9036c54ac141e56bbfe89d7ca5df87",
    "test.question": "213a1d85edf230f46d86005cca6829eb77b07bf802b7a25be0ef44b12c56be81",
    "test.answer": "7a9d7e86c3dc0938bb2d7543394823fca87b8f74620f748ffd8d9cc217bbd3be",
}
WANDB_PROJECT = "codeqa-kc"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def load_split(split):
    """Rows aligned by line index, read exactly like the historical runners (splitlines + strip)."""
    columns = {}
    for ext in ("code", "question", "answer"):
        path = DATA / split / f"{split}.{ext}"
        expected = EXPECTED_SHA256.get(f"{split}.{ext}")
        if expected is not None and sha256_file(path) != expected:
            raise ValueError(f"Hash mismatch for {path}; splits must not change")
        columns[ext] = path.read_text(encoding="utf-8").splitlines()
        if len(columns[ext]) != SPLIT_SIZES[split]:
            raise ValueError(f"{path}: {len(columns[ext])} lines, expected {SPLIT_SIZES[split]}")
    # .strip() per field, exactly as the historical CodeBERT/BART/Seq2seq runners' load_rows.
    rows = [{"id": i, "code": c.strip(), "question": q.strip(), "answer": a.strip()}
            for i, (c, q, a) in enumerate(zip(columns["code"], columns["question"], columns["answer"]))]
    if not all(r["code"] and r["question"] and r["answer"] for r in rows):
        raise ValueError(f"empty field in {split}")
    return rows


def data_hashes():
    return {f"{s}.{e}": sha256_file(DATA / s / f"{s}.{e}")
            for s in SPLIT_SIZES for e in ("code", "question", "answer")}


def codet5_source(question, code, mode):
    """CodeT5 template of the historical FULL run; QONLY removes the code part only."""
    if mode == "full":
        return f"question: {question}\ncode: {code}"
    if mode == "qonly":
        return f"question: {question}\ncode: "
    raise ValueError(mode)


def per_example(predictions, references):
    """Per-example sentence BLEU (official google_bleu), token F1 and EM, all 0-100."""
    bleu = codeqa_metrics._load("bleu/google_bleu.py", str(REPO))
    _, _, individual = bleu.corpus_bleu({i: [p] for i, p in enumerate(predictions)},
                                        {i: [r] for i, r in enumerate(references)})
    out = []
    for i, (prediction, reference) in enumerate(zip(predictions, references)):
        p, r = prediction.lower().split(), reference.lower().split()
        overlap = sum((Counter(p) & Counter(r)).values())
        f1 = 2 * overlap / (len(p) + len(r)) if p or r else 1.0
        out.append({"BLEU": 100.0 * individual[i], "F1": 100.0 * f1, "EM": 100.0 * (p == r)})
    return out


def evaluate(predictions, references, include_meteor=True):
    """Aggregate metrics via codeqa_metrics plus per-example scores; checks they agree."""
    agg = codeqa_metrics.compute_metrics(predictions, references, root=REPO, include_meteor=include_meteor)
    per = per_example(predictions, references)
    n = len(per)
    for key in ("BLEU", "F1", "EM"):
        mean = sum(x[key] for x in per) / n
        if abs(mean - agg[key]) > 1e-6:
            raise AssertionError(f"per-example {key} mean {mean} != aggregate {agg[key]}")
    agg["BLEU_corpus"] = agg["corpus_BLEU_diagnostic"]
    return agg, per


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def write_predictions(path, rows, predictions, per, condition, code_ids):
    """test_predictions.jsonl with id, condition, question, code_id_used, prediction, reference, metrics."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        for row, prediction, scores, code_id in zip(rows, predictions, per, code_ids):
            stream.write(json.dumps({"id": row["id"], "condition": condition, "question": row["question"],
                                     "code_id_used": code_id, "prediction": prediction,
                                     "reference": row["answer"], **scores}, ensure_ascii=False) + "\n")
    tmp.replace(path)


def wandb_env():
    os.environ["WANDB_BASE_URL"] = os.environ.get("WANDB_BASE_URL", "https://api.forge.coreweave.com")
    os.environ["WANDB_ENTITY"] = os.environ.get("WANDB_ENTITY", "chadafihere_kc")
    os.environ.setdefault("WANDB_DISABLE_CODE", "true")
    os.environ.setdefault("WANDB_LOG_MODEL", "false")
    os.environ.setdefault("WANDB_WATCH", "false")


def wandb_init(out_dir, run_tag, group, tags, config, job_type):
    """One W&B run per job; the run id is persisted so an identical-config resume reuses it."""
    wandb_env()
    import wandb
    id_file = Path(out_dir) / "wandb_run.json"
    run_id = json.loads(id_file.read_text())["id"] if id_file.exists() else uuid.uuid4().hex[:8]
    run = wandb.init(project=WANDB_PROJECT, entity=os.environ["WANDB_ENTITY"], name=run_tag, group=group,
                     tags=tags, config=config, job_type=job_type, id=run_id, resume="allow",
                     dir=str(out_dir), settings=wandb.Settings(disable_git=True))
    write_json(id_file, {"id": run.id, "project": WANDB_PROJECT, "entity": os.environ["WANDB_ENTITY"]})
    return run


def wandb_finalize(run, out_dir, run_tag, test_metrics, records, prefix="test"):
    """Summary metrics, a 200-row sample table, and results + predictions as an artifact."""
    import wandb
    run.summary.update({f"{prefix}/{k}": v for k, v in test_metrics.items()
                        if isinstance(v, (int, float)) and not isinstance(v, bool)})
    sample = sorted(random.Random(42).sample(records, min(200, len(records))), key=lambda r: r["id"])
    table = wandb.Table(columns=["id", "condition", "question", "prediction", "reference", "BLEU", "F1"])
    for r in sample:
        table.add_data(r["id"], r["condition"], r["question"], r["prediction"], r["reference"], r["BLEU"], r["F1"])
    run.log({f"{prefix}/sample_predictions": table})
    artifact = wandb.Artifact(name=f"{run_tag}-outputs", type="results")
    for name in ("results.json", "test_predictions.jsonl"):
        if (Path(out_dir) / name).exists():
            artifact.add_file(str(Path(out_dir) / name))
    run.log_artifact(artifact)


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]
