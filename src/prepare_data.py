import os
import json
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

def load_split(data_dir, split):
    split_dir = Path(data_dir) / split
    with open(split_dir / f"{split}.code.original", "r", encoding="utf-8") as f:
        codes_orig = [line.strip() for line in f]
    with open(split_dir / f"{split}.code", "r", encoding="utf-8") as f:
        codes = [line.strip() for line in f]
    with open(split_dir / f"{split}.question", "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f]
    with open(split_dir / f"{split}.answer", "r", encoding="utf-8") as f:
        answers = [line.strip() for line in f]

    assert len(codes) == len(questions) == len(answers), f"Split {split} length mismatch!"
    return {
        "code_original": codes_orig,
        "code": codes,
        "question": questions,
        "answer": answers
    }

def analyze_dataset(data_dir="data/python", tokenizer_name="Qwen/Qwen2.5-Coder-0.5B"):
    print("Loading tokenizer for token distribution analysis...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    splits = ["train", "dev", "test"]
    data = {}
    for s in splits:
        data[s] = load_split(data_dir, s)
        print(f"Split {s}: {len(data[s]['question']} samples")

    # Statistics
    stats = {}
    train_codes = set(data["train"]["code"])
    train_answers = set(data["train"]["answer"])

    for s in splits:
        codes = data[s]["code"]
        answers = data[s]["answer"]
        unique_codes = len(set(codes))

        # Overlap with train
        overlap_count = sum(1 for c in set(codes) if c in train_codes) if s != "train" else unique_codes

        # Answer repetition against train
        repetition_count = sum(1 for a in answers if a in train_answers) if s != "train" else 0
        repetition_rate = (repetition_count / len(answers)) * 100 if len(answers) > 0 else 0.0

        # Span / extraction rate (answer is substring of code)
        span_count = 0
        exact_token_in_code_count = 0
        for code, ans in zip(codes, answers):
            if ans.lower() in code.lower():
                span_count += 1
            # Check if all tokens in answer appear in code
            ans_tokens = set(ans.lower().split())
            code_tokens = set(code.lower().split())
            if ans_tokens.issubset(code_tokens):
                exact_token_in_code_count += 1

        span_rate = (span_count / len(answers)) * 100
        token_match_rate = (exact_token_in_code_count / len(answers)) * 100

        stats[s] = {
            "total_samples": len(codes),
            "unique_codes": unique_codes,
            "code_overlap_with_train_pct": (overlap_count / unique_codes * 100) if unique_codes > 0 else 100.0,
            "answer_repetition_pct": repetition_rate,
            "span_rate_pct": span_rate,
            "token_match_pct": token_match_rate
        }

    print("\nDataset Statistics:")
    print(json.dumps(stats, indent=2))

    # Token length distribution on train set
    train_code_lens = []
    train_q_lens = []
    train_a_lens = []
    train_total_lens = []

    for code, q, a in zip(data["train"]["code"], data["train"]["question"], data["train"]["answer"]):
        prompt = f"### Code\n{code}\n### Question\n{q}\n### Answer\n{a}<|endoftext|>"
        tokens = tokenizer(prompt, truncation=False)["input_ids"]
        train_total_lens.append(len(tokens))
        train_code_lens.append(len(tokenizer(code)["input_ids"]))
        train_q_lens.append(len(tokenizer(q)["input_ids"]))
        train_a_lens.append(len(tokenizer(a)["input_ids"]))

    p99_total = int(np.percentile(train_total_lens, 99))
    # Round to multiple of 64
    max_length = int(np.ceil(p99_total / 64) * 64)
    print(f"\nTrain set token lengths - P99 total: {p99_total}, chosen max_length (multiple of 64): {max_length}")

    os.makedirs("runs/metrics", exist_ok=True)
    with open("runs/metrics/dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "max_length": max_length, "p99_total": p99_total}, f, indent=2)

    # Save processed splits as jsonl
    os.makedirs("runs/processed", exist_ok=True)
    for s in splits:
        out_path = f"runs/processed/{s}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for c, co, q, a in zip(data[s]["code"], data[s]["code_original"], data[s]["question"], data[s]["answer"]):
                item = {
                    "code": c,
                    "code_original": co,
                    "question": q,
                    "answer": a
                }
                f.write(json.dumps(item) + "\n")
        print(f"Saved processed {s} to {out_path}")

if __name__ == "__main__":
    analyze_dataset()
