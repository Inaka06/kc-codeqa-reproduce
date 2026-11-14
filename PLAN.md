# Implementation Plan and Time Estimates

**Project**: Retrieval-Augmented Code QA (FiD + REDCODER + CodeQA Python)

**Start Date**: 2026-11-11  
**Time Budget**: 5 days maximum  
**Hardware**: RTX 5050 Laptop GPU, 8GB VRAM (constrained)

---

## Critical Path Summary

| Day | Main Task | Hours | Deliverable |
|-----|-----------|-------|-------------|
| 1 | Setup + Data Prep | 8 | CodeQA loaded, CSN normalized |
| 2 | Retriever Training | 10 | SCODE-R trained, FAISS index, offline retrieval |
| 3 | A0 Baseline | 12 | No-retrieval baseline trained |
| 4 | A2 FiD Main | 14 | FiD model trained (primary contribution) |
| 5 | A1 + Ablations | 12 | REDCODER concat, ablation studies |

**Total**: ~56-60 hours across 5 days

---

## Phase 1: Setup and Data Preparation (Day 1)

### ✅ 1.1 Environment Setup (1 hour)
- Install PyTorch with CUDA support
- Install dependencies from requirements.txt
- Verify RTX 5050 8GB GPU accessible

### 1.2 Download CodeQA Dataset (1 hour)
- Clone https://github.com/jadecxliu/CodeQA
- Extract Python dataset
- **Verify**: 56,085 train / 7,000 dev / 7,000 test
- Analyze format and statistics

### 1.3 Download and Normalize CodeSearchNet Python (3 hours)
- Download CSN Python from HuggingFace datasets
- Normalize code: remove docstrings, replace `STRING`/`_NUM`
- Extract docstring first sentence as summary
- Deduplicate by code hash
- **Target**: ~400k-500k functions

### 1.4 Leakage Prevention (2 hours)
- Hash CodeQA codes (train/dev/test)
- Remove exact matches from CSN
- Remove near-duplicates (Jaccard ≥ 0.9)
- **Report**: CSN original → dedup → after leakage
- **Target**: 200k-400k final database

### 1.5 Smoke Test (1 hour)
- Test data loading on 100 samples
- Verify metrics (BLEU, ROUGE-L, METEOR, EM, F1)

---

## Phase 2: Retriever Training (Day 2)

### 2.1 Train SCODE-R Dense Retriever (6 hours)
- **Model**: Bi-encoder from `microsoft/graphcodebert-base`
- **Data**: (code, docstring) pairs from CSN
- **Config**:
  - Query max length: 256, doc max length: 64
  - Batch size: 16-32, bf16, gradient checkpointing
  - LR: 2e-5, warmup 10%, 3-5 epochs
  - Loss: DPR with in-batch negatives only
- **Eval**: MRR, Recall@1/5/10
- **Time estimate**: 4-5 hours on 8GB GPU

### 2.2 Build FAISS Index (1 hour)
- Encode database with document encoder
- Build IndexFlatIP (inner product)
- Save to disk

### 2.3 BM25 Baseline (1 hour)
- Implement BM25 with rank_bm25
- Compare Recall@K with SCODE-R

### 2.4 Offline Retrieval (2 hours)
- Retrieve top-k=10 for all CodeQA samples
- Save to `data/retrieved_{split}.jsonl`
- Format: `{question, code, answer, retrieved: [{code, docstring, score}]}`

---

## Phase 3: Generator Training - Baseline A0 (Day 3)

### 3.1 No Retrieval Baseline (A0) (10 hours)
- **Model**: CodeT5-base (`Salesforce/codet5-base`)
- **Input**: `question: {q} context: {code}`
- **Config**:
  - Max input: 512, max output: 32
  - Batch per GPU: 4, grad accum: 8 → effective 32
  - bf16, gradient checkpointing
  - LR: 1e-4 (or 5e-5), AdamW, weight decay 0.01
  - Warmup 10%, linear decay
  - Max epochs: 15, patience: 3 on dev BLEU
  - Eval every epoch, beam 4
- **Time estimate**: ~8-10 hours (15-20 min/epoch × ~10 epochs)
- **Expected**: BLEU ~30-35, F1 ~25-32

### 3.2 Evaluate A0 (2 hours)
- Generate on full dev (7,000) and test (7,000)
- Compute all metrics
- **Sanity check**: F1 ≥ 25, else debug before continuing

---

## Phase 4: FiD Model Training (Day 4)

### 4.1 Implement FiD Architecture (2 hours)
- Modify CodeT5 forward for FiD:
  - Input: `[batch, k, L]` → `[batch*k, L]`
  - Encode independently
  - Concat for decoder: `[batch, k*L, d]`
- Test with k=5, gradient checkpointing

### 4.2 Train A2: FiD (k=5) (10 hours)
- **Input per passage**: `question: {q} code: {code} context: {retrieved_code} summary: {retrieved_docstring}`
- **Config**:
  - Max length per passage: 256, k=5
  - Batch per GPU: 2, grad accum: 16 → effective 32
  - Other params same as A0
- **Time estimate**: ~10-12 hours (4-5× slower than A0)

### 4.3 Evaluate A2 (2 hours)
- Generate on dev/test (k=5)
- Compare with A0

---

## Phase 5: Additional Experiments (Day 5)

### 5.1 Train A1: REDCODER Concat (6 hours)
- **Input**: `{q} [csep] {code} [csep] {Y1} [nsep] {X1} ...`
- k=3-5, truncate to 512
- Same config as A0
- **Time**: ~5-6 hours

### 5.2 Ablation A4: Passage Content (3 hours)
- A4a: Retrieved code only
- A4b: Docstring only
- A4c: Both (A2, done)
- **Time**: ~1-1.5 hours each (fine-tune from A2)

### 5.3 Ablation A5: BM25 vs Dense (2 hours)
- Use BM25 passages with FiD
- Compare retrieval impact

### 5.4 Optional A3: k=10 Evaluation (1 hour)
- Evaluate A2 with k=10 (if time permits)

---

## Final Phase: Evaluation and Reporting

### Generate REPORT.md
- Results table (all models, dev/test)
- Paper baselines for comparison
- Retriever evaluation
- Leakage statistics
- Training curves
- Ablation analysis
- Qualitative examples (≥10)
- Error analysis
- Limitations and deviations

### Generate README.md
- Step-by-step reproduction
- Environment setup
- Training commands
- Evaluation commands

---

## Priority if Time-Constrained

1. ✅ **A0** (no retrieval) - MUST HAVE
2. ✅ **A2** (FiD main) - MUST HAVE
3. ⚠️ **A1** (REDCODER) - SHOULD HAVE
4. ⚠️ **A4** (ablation) - NICE TO HAVE
5. ❌ **A3, A5** - SKIP IF NEEDED

---

## Memory Optimization Strategy

**If OOM during training**:
1. Reduce batch size to 1-2
2. Increase gradient accumulation
3. Reduce k from 5 to 3
4. Reduce passage length from 256 to 192
5. Use 8-bit optimizer (bitsandbytes)
6. Train retriever on CSN subset

**8GB VRAM is tight** - expect to use smallest viable batch sizes.

---

## Checkpoints and Resume

- Save optimizer state, scheduler, RNG for resume
- Keep 2 best checkpoints only (delete intermediate)
- Use `nohup` or `tmux` for long runs
- Monitor logs regularly for NaN/divergence

---

## Next Steps

1. ✅ NOTES.md created with paper analysis
2. ✅ PLAN.md created with timeline
3. ⏭️ Install Python environment
4. ⏭️ Download CodeQA
5. ⏭️ Download and normalize CSN Python

**Status**: Ready to begin Phase 1 setup
