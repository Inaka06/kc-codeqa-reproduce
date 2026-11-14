"""
Metrik evaluasi CodeQA: BLEU, ROUGE-L, METEOR, Exact Match (EM), F1.
Satu implementasi dipakai untuk semua model agar perbandingan konsisten.
Semua skor dalam skala 0-100.

Catatan: detail implementasi (tokenisasi BLEU, beta ROUGE-L, dst.) bisa sedikit
berbeda dari kode evaluasi paper CodeQA, jadi bandingkan tren antar model,
bukan selisih desimal terhadap angka di paper.
"""
import re
import string
from collections import Counter

_PUNCT = set(string.punctuation)


def normalize_answer(s):
    """Lowercase, hapus tanda baca dan artikel, rapikan spasi (ala SQuAD)."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred, ref):
    return float(normalize_answer(pred) == normalize_answer(ref))


def f1_score(pred, ref):
    p = normalize_answer(pred).split()
    r = normalize_answer(ref).split()
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0
    overlap = sum((Counter(p) & Counter(r)).values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(r)
    return 2 * prec * rec / (prec + rec)


def _lcs_len(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(pred, ref):
    """ROUGE-L F-measure (beta=1) pada token whitespace berhuruf kecil."""
    p, r = pred.lower().split(), ref.lower().split()
    if not p or not r:
        return 0.0
    lcs = _lcs_len(p, r)
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(p), lcs / len(r)
    return 2 * prec * rec / (prec + rec)


_meteor_fn = None
_meteor_ready = None


def _init_meteor():
    global _meteor_fn, _meteor_ready
    if _meteor_ready is not None:
        return _meteor_ready
    try:
        import nltk
        from nltk.translate.meteor_score import meteor_score

        try:
            meteor_score([["good", "dog"]], ["bad", "cat"])  # kata beda -> memicu tahap WordNet
        except LookupError:
            for pkg in ("wordnet", "omw-1.4"):
                try:
                    nltk.download(pkg, quiet=True)
                except Exception:
                    pass
            meteor_score([["good", "dog"]], ["bad", "cat"])  # kata beda -> memicu tahap WordNet
        _meteor_fn = meteor_score
        _meteor_ready = True
    except Exception as e:  # wordnet tidak ada / tidak ada internet
        print(f"[WARN] METEOR tidak tersedia ({type(e).__name__}: {e}). "
              f"Unduh wordnet NLTK secara manual. METEOR diisi None.")
        _meteor_ready = False
    return _meteor_ready


def meteor(pred, ref):
    p, r = pred.lower().split(), ref.lower().split()
    if not p or not r:
        return 0.0
    return float(_meteor_fn([r], p))


def compute_all(preds, refs):
    """Hitung semua metrik. preds dan refs: list string dengan panjang sama."""
    import sacrebleu

    assert len(preds) == len(refs) and len(preds) > 0
    n = len(preds)
    out = {
        "bleu": sacrebleu.corpus_bleu(preds, [refs]).score,
        "bleu_sent_avg": sum(sacrebleu.sentence_bleu(p, [r]).score for p, r in zip(preds, refs)) / n,
        "rouge_l": 100 * sum(rouge_l(p, r) for p, r in zip(preds, refs)) / n,
        "em": 100 * sum(exact_match(p, r) for p, r in zip(preds, refs)) / n,
        "f1": 100 * sum(f1_score(p, r) for p, r in zip(preds, refs)) / n,
        "meteor": None,
    }
    if _init_meteor():
        out["meteor"] = 100 * sum(meteor(p, r) for p, r in zip(preds, refs)) / n
    return out


if __name__ == "__main__":
    preds = ["the given child", "a list of tables", "page arguments from the request", ""]
    refs = ["The given child.", "a list of simple tables", "page arguments", "how many marks"]
    for k, v in compute_all(preds, refs).items():
        print(f"{k:>14}: {v}")
    assert exact_match("The given child.", "the given child") == 1.0
    assert f1_score("", "") == 1.0 and f1_score("a b", "") == 0.0
    print("self-test OK")