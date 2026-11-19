"""Shared CodeQA metric contract; preserves the historical evaluator definitions."""
from collections import Counter
from functools import lru_cache
import hashlib
import importlib.util
import math
from pathlib import Path
import shutil
import threading


ROOT = Path(__file__).resolve().parent
METRIC_NAMES = ("BLEU", "ROUGE_L", "METEOR", "EM", "Precision", "Recall", "F1")
VERSION = "codeqa-standard-v1"


def _aligned(predictions, references):
    if not references or len(predictions) != len(references):
        raise ValueError("Predictions/references must be nonempty and aligned")
    if not all(isinstance(text, str) for text in [*predictions, *references]):
        raise TypeError("Predictions/references must contain strings")


def answer_metrics(predictions, references):
    """Macro-average token P/R/F1 with lowercase + whitespace, preserving duplicates."""
    _aligned(predictions, references)
    exact, precisions, recalls, f1s, yes_no, non_yes_no = [], [], [], [], [], []
    for prediction, reference in zip(predictions, references):
        p, r = prediction.lower().split(), reference.lower().split()
        overlap = sum((Counter(p) & Counter(r)).values())
        precision = overlap / len(p) if p else float(not r)
        recall = overlap / len(r) if r else float(not p)
        f1 = 2 * overlap / (len(p) + len(r)) if p or r else 1.0
        exact.append(p == r)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        (yes_no if reference.strip().lower() in ("yes", "no") else non_yes_no).append(p == r)
    mean = lambda values: 100 * sum(values) / len(values)
    return {"n": len(references), "EM": mean(exact), "Precision": mean(precisions),
        "Recall": mean(recalls), "F1": mean(f1s), "token_f1": mean(f1s),
        "yes_no_correct": sum(yes_no), "yes_no_n": len(yes_no),
        "non_yes_no_correct": sum(non_yes_no), "non_yes_no_n": len(non_yes_no),
        "yes_no_EM": mean(yes_no) if yes_no else None,
        "non_yes_no_EM": mean(non_yes_no) if non_yes_no else None}


@lru_cache(maxsize=8)
def _load(relative, root):
    path = Path(root) / "source/codeBERT/code/eval" / relative
    if not path.is_file():
        raise FileNotFoundError(f"Required CodeQA evaluator unavailable: {path}")
    spec = importlib.util.spec_from_file_location("codeqa_standard_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _meteor(hypotheses, references, root):
    path = Path(root) / "source/codeBERT/code/eval/meteor/meteor.py"
    for required in (path, path.with_name("meteor-1.5.jar"), path.parent / "data/paraphrase-en.gz"):
        if not required.is_file():
            raise FileNotFoundError(f"METEOR resource missing: {required}")
    if shutil.which("java") is None:
        raise RuntimeError("METEOR 1.5 requires Java; do not substitute NLTK METEOR")
    # Same Java argument-order correction used by the historical CodeBERT/Seq2seq runners.
    text = path.read_text(encoding="utf-8").replace(
        "['java', '-jar', '-Xmx{}'.format(mem), METEOR_JAR,",
        "['java', '-Xmx{}'.format(mem), '-jar', METEOR_JAR,")
    namespace = {"__file__": str(path)}
    exec(compile(text, str(path), "exec"), namespace)
    meteor = namespace["Meteor"]()
    process = meteor.meteor_p
    timed_out = threading.Event()

    def terminate():
        if process.poll() is None:
            timed_out.set()
            process.kill()

    watchdog = threading.Timer(120, terminate)
    watchdog.daemon = True
    watchdog.start()
    try:
        return float(100 * meteor.compute_score(references, hypotheses)[0])
    except Exception as exc:
        if timed_out.is_set():
            raise TimeoutError("METEOR exceeded 120 seconds") from exc
        raise RuntimeError("Official METEOR evaluation failed; check Java/resources") from exc
    finally:
        watchdog.cancel()
        meteor.close()
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()


def compute_metrics(predictions, references, *, root=ROOT, include_meteor=True):
    """All scores use 0–100. BLEU is mean smoothed sentence BLEU, NOT corpus BLEU."""
    result = answer_metrics(predictions, references)
    if any(not reference.strip() for reference in references):
        raise ValueError("Generation metrics require nonempty reference answers")
    hypotheses = {i: [text] for i, text in enumerate(predictions)}
    gold = {i: [text] for i, text in enumerate(references)}
    corpus, sentence, _ = _load("bleu/google_bleu.py", str(root)).corpus_bleu(hypotheses, gold)
    rouge, _ = _load("rouge/rouge.py", str(root)).Rouge().compute_score(gold, hypotheses)
    result.update(BLEU=float(100 * sentence), ROUGE_L=float(100 * rouge),
        corpus_BLEU_diagnostic=float(100 * corpus),
        BLEU_sentence_smoothed=float(100 * sentence), BLEU_corpus_smoothed=float(100 * corpus))
    if include_meteor:
        result["METEOR"] = _meteor(hypotheses, gold, root)
    for name in METRIC_NAMES:
        if name in result and not math.isfinite(result[name]):
            raise ValueError(f"Non-finite metric: {name}")
    return result


@lru_cache(maxsize=8)
def metric_provenance(root=ROOT):
    if shutil.which("java") is None:
        raise RuntimeError("Official METEOR requires Java before starting this experiment")
    paths = [Path(__file__), *[Path(root) / "source/codeBERT/code/eval" / relative for relative in (
        "bleu/google_bleu.py", "rouge/rouge.py", "meteor/meteor.py", "meteor/meteor-1.5.jar",
        "meteor/data/paraphrase-en.gz")]]
    hashes = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required evaluator resource missing: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for part in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(part)
        hashes[str(path.relative_to(root))] = digest.hexdigest()
    return {"version": VERSION, "scale": "0-100", "BLEU": "mean_sentence_bleu4_add_one_smoothed",
            "token_normalization": "lowercase_whitespace", "generation_text": "original_case_punctuation",
            "METEOR": "official_meteor_1.5_en_norm", "sha256": hashes}
