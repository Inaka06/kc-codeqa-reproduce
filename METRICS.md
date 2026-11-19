# CodeQA evaluation contract — all new experiments

Use `codeqa_metrics.compute_metrics(predictions, references)`. Do not replace these metrics with similarly named implementations. All scores are on a **0–100** scale; full references, never truncated training labels, are the gold for scoring.

| Canonical key | Exact definition | Frequency |
|---|---|---|
| `BLEU` | Mean **sentence** BLEU-4, add-one smoothing; `source/codeBERT/code/eval/bleu/google_bleu.py` | Each dev epoch + final test |
| `ROUGE_L` | Official local token LCS scorer, beta=1.2 | Each dev epoch + final test |
| `METEOR` | Official Java METEOR 1.5, English, `-norm`, bundled paraphrase resource | Each dev epoch + final test |
| `EM` | Lowercase + whitespace normalization; punctuation retained | Each dev epoch + final test |
| `Precision`, `Recall`, `F1` | Macro average of per-answer token scores; multiset overlap, same normalization | Each dev epoch + final test |

Also retain `corpus_BLEU_diagnostic` (same smoothing, aggregate corpus; **not** the main BLEU), yes/no and non-yes/no counts/EM, train/dev loss, LR, epoch/step, sample counts and split scope. `token_f1`, `BLEU_sentence_smoothed` and `BLEU_corpus_smoothed` are compatibility aliases, not different metrics.

W&B names: `eval/BLEU`, `eval/ROUGE_L`, `eval/METEOR`, `eval/EM`, `eval/Precision`, `eval/Recall`, `eval/F1`, and corresponding `test/*` keys. Dev metrics are generated without teacher forcing; test only after model selection. Score all metrics on the **same predictions**, not another decoding run per metric. Missing Java/resources must fail clearly, never silently use zero or an unrelated NLTK implementation. Record evaluator/resource hashes via `metric_provenance()`.

## Audit evidence

- VPS Seq2seq `seq2seq_paper_results/paper_v1_0afa6482d5df/results.json`: BLEU 29.0751, ROUGE-L 20.5459, METEOR 7.0397, EM 3.0000, F1 23.8160 (7,000 test).
- VPS CodeBERT–Transformer `codebert_paper_results/paper_v1_67d4ca132a23/results.json`: BLEU 34.6109, ROUGE-L 29.7997, METEOR 11.9331, EM 4.9714, F1 31.1732 (7,000 test).
- Those runners compute BLEU/ROUGE/EM/F1 on dev, METEOR at final test; both select checkpoints by **Dev BLEU**. The official `source/codeBERT/code/run.py::eval_accuracies` additionally returns Precision/Recall. New experiments report all seven on dev and test to avoid omissions.
- The separate Moses/CodeBERT BLEU path in upstream `run.py` is **not** the main `eval_accuracies` BLEU; do not mix the numbers.

## Comparability and model selection

Declare the dev selection metric before training. The default for new runs is **Dev BLEU**, matching the historical baselines; log all the other metrics as well. Never select using test. The existing CodeT5 pilot was selected by Dev Token F1; adding metrics afterward must not silently relabel/reselect it.

Matching metric definitions is necessary but not sufficient: record train/dev/test IDs, dataset/preprocessing hashes, decoding settings, parameter count/pretraining, seed, limits and runtime. A 500-example pilot is not directly comparable to the historical 7,000-example tests.

Do not fabricate missing historical epoch curves. A post-hoc best-checkpoint evaluation gives a **single** dev point, explicitly labeled post-hoc. Preserve original result files and selection criteria; write supplementary metrics separately.
