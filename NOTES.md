# Technical Notes from Three Papers

## 1. CodeQA (Liu & Wan, EMNLP Findings 2021)

### Task Definition
- **Goal**: Free-form question answering for source code comprehension
- **Input**: Code snippet `c` + natural language question `q`
- **Output**: Free-form textual answer `a` (word, phrase, or sentence)
- **Key challenge**: Answer usually cannot be directly extracted from code

### Dataset Statistics (Python)
- **Train**: 56,085 QA pairs from 34,641 unique codes
- **Dev**: 7,000 QA pairs
- **Test**: 7,000 QA pairs
- **Token statistics**:
  - Code: avg 48.97 tokens (±49 tokens as mentioned in prompt)
  - Question: avg 8.15 tokens (±8 tokens)
  - Answer: avg 4.07 tokens (±4 tokens)
- **Unique tokens**: 108,571 in code, 11,401 in questions, 12,723 in answers
- **Answer repetition rate**: Dev 1.17%, Test 1.56% (against train)
- **Span extraction rate**: 4.99%
- **Question types**: 67.24% What, 27.77% How/Where/When/Why/Purpose, 2.86% Yes/No

### Data Construction
- **Source**: Python docstrings from GitHub (Barone & Sennrich, 2017)
- **Method**: Syntactic rules + semantic analysis to transform comments into QA pairs
- **Preprocessing**: Code has tokens like `_NUM` and `STRING` replacing literals, no docstrings/comments
- **Split**: 8:1:1 ratio after shuffling pairs (NOT per-code split)
- **CRITICAL**: Same code can appear in train and test with different questions

### Baseline Results (Python) - Table 8
**Development Set**:
| Model | BLEU | ROUGE-L | METEOR | EM | F1 |
|-------|------|---------|--------|----|----|
| Seq2seq | 29.70 | 22.18 | 6.36 | 2.95 | 23.67 |
| Dual Encoder | 28.57 | 17.62 | 4.43 | 2.22 | 18.78 |
| Transformer | 30.85 | 23.96 | 7.91 | 3.37 | 25.30 |
| CodeBERT | 33.84 | 30.27 | 11.51 | 5.50 | 31.57 |

**Test Set**:
| Model | BLEU | ROUGE-L | METEOR | EM | F1 |
|-------|------|---------|--------|----|----|
| Seq2seq | 30.09 | 21.69 | 6.38 | 2.84 | 23.27 |
| Dual Encoder | 28.90 | 17.79 | 4.53 | 2.22 | 18.97 |
| Transformer | 30.69 | 23.26 | 7.80 | 3.35 | 24.59 |
| CodeBERT | 34.86 | 30.28 | 12.51 | 4.93 | 31.56 |

### Baseline Hyperparameters (Appendix B)
**Seq2seq**:
- Bidirectional RNN, 2 layers, hidden size 512
- Batch size 32, beam size 4, LR 0.002, dropout 0.2
- Adam optimizer
- Max 200 epochs, early stop patience 20

**Dual Encoder**:
- Same as Seq2seq but with two encoders (code-info + question-info)

**Transformer**:
- 6 layers, 8 heads, hidden size 512
- Batch size 32, beam size 4, initial LR 0.0001, dropout 0.2
- Relative position representations + copy attention
- Max 200 epochs, early stop patience 20

**CodeBERT**:
- Encoder: pre-trained CodeBERT
- Decoder: Transformer 6 layers, 12 heads
- Batch size 64, beam size 10, LR 5e-5
- Adam optimizer
- 20 epochs on 3× Nvidia 1080 Ti GPUs
- Select checkpoint with best BLEU score

### Evaluation Metrics
- **BLEU**: Corpus-level BLEU (Papineni et al., 2002)
- **ROUGE-L**: Longest common subsequence F-score
- **METEOR**: With stemming and synonyms
- **EM (Exact Match)**: After normalization (lowercase, remove articles/punctuation)
- **F1**: Token-level F1 like SQuAD (after normalization)

---

## 2. Fusion-in-Decoder / FiD (Izacard & Grave, 2021)

### Core Architecture
- **Two-stage**: (1) Retrieve passages with BM25/DPR, (2) Generate answer with T5/BART
- **Key innovation**: Process passages **independently** in encoder, jointly in decoder
- **Input format per passage**: `question: {q} title: {title} context: {passage_text}`
- **Encoder**: Each (question + passage) pair encoded separately → `[batch, k, L]` reshaped to `[batch*k, L]`
- **Decoder**: Attends over **concatenation** of all k encoder outputs → `[batch, k*L, d]`
- **Complexity**: Linear in k (not quadratic), because encoder self-attention is per-passage

### Performance Benefits
- **Scales with k**: NaturalQuestions improves 3.5% from k=10 to k=100
- **TriviaQA**: Improves 6% from k=10 to k=100
- **Best results**: 51.4% EM on NQ, 67.6% on TriviaQA with T5-large + DPR

### Training Strategy
- **Progressive training**: Train with fewer passages (e.g., k=5), then fine-tune with more (k=100) for 1000 steps
- **Saves compute**: 147 GPU hours vs 425 GPU hours for full k=100 training
- **T5-base**: 220M params, T5-large: 770M params

### Hyperparameters (from paper)
- **Model**: T5-base or T5-large (HuggingFace Transformers)
- **Optimizer**: Adam, constant LR 1e-4, dropout 10%
- **Training**: 10k gradient steps, batch size 64
- **Evaluation**: Every 500 steps, select best on validation EM
- **Retrieval**: 100 passages (unless stated), truncated to 250 word pieces
- **Decoding**: Greedy
- **Hardware**: 64× Tesla V100 32GB GPUs

---

## 3. REDCODER (Parvez et al., 2021)

### Framework
- **Two-component**: (1) Retriever (SCODE-R), (2) Generator (SCODE-G)
- **Tasks**: Code generation and code summarization
- **Key idea**: Augment input with retrieved code/summaries from real repositories

### Retriever: SCODE-R
- **Architecture**: Bi-encoder based on DPR (Karpukhin et al., 2020)
- **Encoders**: Two separate encoders initialized from **GraphCodeBERT** (or CodeBERT)
  - Query encoder: for code (in code→summary direction) or summary (in summary→code)
  - Document encoder: for summary or code
- **Similarity**: Dot product of `[CLS]` representations
- **Loss**: DPR loss with **in-batch negatives only** (NO hard negatives)
  - Reason: Retrieved code/summary that doesn't match target can still help generation
- **Training**: Fine-tune on parallel (code, docstring) pairs from CodeSearchNet
- **Index**: FAISS with IndexFlatIP (inner product search)

### Generator: SCODE-G
- **Base model**: PLBART (Ahmad et al., 2021)
- **Input augmentation**:
  - **REDCODER (singleton)**: `x [csep] Y1 [csep] Y2 ... [csep] Yk`
  - **REDCODER-EXT (pairs)**: `x [csep] Y1 [nsep] X1 [csep] Y2 [nsep] X2 ...`
    - Yj = retrieved code/summary, Xj = its pair (if available)
- **Max input length**: 512 tokens (PLBART limit), truncate as needed

### Results on CodeXGLUE Python Code Generation
| Method | EM | BLEU | CodeBLEU |
|--------|-----|------|----------|
| BM25 (retrieval only) | 0.00 | 6.63 | 13.49 |
| SCODE-R (retrieval only) | 0.00 | 22.75 | 23.92 |
| PLBART (no retrieval) | 0.00 | 4.89 | 12.01 |
| BM25 + PLBART | 0.03 | 6.99 | 13.89 |
| REDCODER | 8.88 | 22.74 | 28.93 |
| REDCODER-EXT | 9.61 | 24.43 | 30.21 |

### Results on Code Summarization (BLEU-4)
| Method | Python | Java |
|--------|--------|------|
| BM25 | 1.92 | 1.82 |
| SCODE-R | 14.98 | 15.87 |
| PLBART | 19.30 | 18.45 |
| BM25 + PLBART | 19.57 | 19.71 |
| REDCODER | 21.01 | 22.94 |
| REDCODER-EXT | 20.91 | 22.95 |

### Hyperparameters (from Section 4.1)
- **SCODE-R**: GraphCodeBERT encoders, fine-tuned with DPR loss
- **SCODE-G (PLBART)**:
  - Max epoch: 15
  - Patience: 5 (early stopping)
  - Learning rate: 2e-5
  - Batch size: tuned in {8, 16, 32, 64, 72}
  - k (top-k retrieval): up to 10 for code gen, {10, 30, 50, 100} for summarization
  - Max input length: 512

### Retrieval Database
- **Source**: CodeSearchNet (Husain et al., 2019)
- **Code database**: ~1.4M functions (Java + Python), 59% paired, 41% monolingual
- **Summary database**: ~1.1M unique summaries from CSN + CCSD corpus
- **Deduplication**: Applied to both code and summaries
- **Target exclusion**: Target code/summary removed from retrieval database by default

---

## Key Design Decisions for This Project

### 1. Why No Hard Negatives for Retriever?
- REDCODER paper (Section 3.1, Figure 3): Unlike open-domain QA where hard negatives help distinguish target from similar passages, for code tasks, a retrieved code that doesn't match the target can still benefit generation
- We follow REDCODER: train SCODE-R with **in-batch negatives only**

### 2. Input Format
- **A0 (no retrieval)**: `question: {q} context: {code}`
- **A1 (REDCODER)**: `{q} [csep] {code} [csep] {Y1} [nsep] {X1} ...` (truncate to 512)
- **A2 (FiD)**: Per passage: `question: {q} code: {code} context: {retrieved_code} summary: {retrieved_docstring}`
  - Each passage encoded independently, max length 256 per passage
  - Decoder attends over concatenated representations

### 3. Why CodeT5-base for Generator?
- T5-style architecture (most similar to FiD paper's T5)
- ~220M params (similar to T5-base)
- Pre-trained on code (CodeSearchNet + others)
- PLBART-base as alternative if time permits

### 4. Expected Baseline Performance
- **A0 (no retrieval)** should achieve BLEU ~30-35, F1 ~25-32 (CodeBERT range)
- If A0 achieves F1 < 25, investigate bugs before proceeding to retrieval experiments

### 5. Leakage Prevention
- CodeQA answers derived from code comments/docstrings
- Must remove from database any CSN function that matches (or nearly matches via Jaccard ≥ 0.9) any CodeQA train/dev/test code
- Report number of entries removed

### 6. Evaluation Consistency
- Use **same metric implementation** for all models
- BLEU: corpus-level, specify smoothing method
- F1: token-level after normalization (lowercase, remove articles/punctuation)
- EM: exact match after same normalization
- Checkpoint selection: based on **BLEU on dev** (consistent with CodeQA paper)

---

## Hardware Constraints

**Available**: RTX 5050 Laptop GPU with **8GB VRAM** (not RTX 3060 12GB as originally stated)

**Memory Optimization Required**:
1. Use **bf16** for T5-style models (FiD, A2)
2. Use **gradient checkpointing** on encoder
3. Start with **small batch size** (2-4 per GPU) + high gradient accumulation (8-16)
4. **Reduce k** if needed (start with k=3-5 for training, evaluate with k=5 or k=10)
5. Consider **8-bit optimizer** (bitsandbytes) if OOM
6. For retriever training: smaller batch size, fp16/bf16
7. **Prioritize experiments**: A0 → A2 → A1 → A4 → A3 → A5

---

## Next Steps

1. ✅ Read all three papers and extract technical details
2. Install Python environment (PyTorch, transformers, datasets, faiss, etc.)
3. Download CodeQA dataset, verify statistics
4. Download and normalize CodeSearchNet Python, prevent leakage
5. Train retriever (SCODE-R) + build FAISS index + BM25 baseline
6. Perform offline retrieval (save top-k=10 for all splits)
7. Train generators: A0 → A2 (main) → A1 → ablations
8. Evaluate all models on dev and test (full sets)
9. Generate comprehensive REPORT.md with results, analysis, and examples
