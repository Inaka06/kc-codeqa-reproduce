"""
Download and prepare CodeQA dataset (Python only)
"""
import os
import json
import subprocess
from pathlib import Path

def download_codeqa():
    """Download CodeQA dataset from GitHub"""
    data_dir = Path("data/codeqa")
    data_dir.mkdir(parents=True, exist_ok=True)

    repo_url = "https://github.com/jadecxliu/CodeQA.git"
    clone_dir = data_dir / "CodeQA_repo"

    if not clone_dir.exists():
        print(f"Cloning CodeQA repository from {repo_url}...")
        subprocess.run(["git", "clone", repo_url, str(clone_dir)], check=True)
    else:
        print(f"CodeQA repository already exists at {clone_dir}")

    return clone_dir

def load_codeqa_split(split_file):
    """Load a CodeQA split file"""
    data = []
    with open(split_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def verify_codeqa_stats(train_data, dev_data, test_data):
    """Verify dataset statistics match paper"""
    print("\n=== CodeQA Dataset Statistics ===")
    print(f"Train size: {len(train_data)} (expected 56,085)")
    print(f"Dev size: {len(dev_data)} (expected 7,000)")
    print(f"Test size: {len(test_data)} (expected 7,000)")

    # Compute token statistics
    def get_stats(data):
        code_lens = [len(item['code'].split()) for item in data]
        q_lens = [len(item['question'].split()) for item in data]
        a_lens = [len(item['answer'].split()) for item in data]
        return {
            'code': sum(code_lens) / len(code_lens),
            'question': sum(q_lens) / len(q_lens),
            'answer': sum(a_lens) / len(a_lens)
        }

    train_stats = get_stats(train_data)
    print(f"\nAverage token lengths (train):")
    print(f"  Code: {train_stats['code']:.2f} (expected ~49)")
    print(f"  Question: {train_stats['question']:.2f} (expected ~8)")
    print(f"  Answer: {train_stats['answer']:.2f} (expected ~4)")

    # Check unique codes
    unique_codes = len(set(item['code'] for item in train_data))
    print(f"\nUnique codes in train: {unique_codes} (expected ~34,641)")

    return True

def prepare_codeqa():
    """Main function to download and prepare CodeQA"""
    clone_dir = download_codeqa()

    # Find Python dataset files
    python_dir = clone_dir / "dataset" / "python"
    if not python_dir.exists():
        print(f"Error: Python dataset not found at {python_dir}")
        print("Please check the repository structure")
        return None

    # Load splits
    train_file = python_dir / "train.jsonl"
    dev_file = python_dir / "dev.jsonl"
    test_file = python_dir / "test.jsonl"

    if not all(f.exists() for f in [train_file, dev_file, test_file]):
        print("Error: Split files not found. Checking alternative locations...")
        for alt_path in [clone_dir / "python", clone_dir / "data" / "python"]:
            if (alt_path / "train.jsonl").exists():
                python_dir = alt_path
                train_file = python_dir / "train.jsonl"
                dev_file = python_dir / "dev.jsonl"
                test_file = python_dir / "test.jsonl"
                break

    print(f"\nLoading CodeQA Python splits from {python_dir}...")
    train_data = load_codeqa_split(train_file)
    dev_data = load_codeqa_split(dev_file)
    test_data = load_codeqa_split(test_file)

    # Verify statistics
    verify_codeqa_stats(train_data, dev_data, test_data)

    # Save to standardized location
    output_dir = Path("data/codeqa_processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in [("train", train_data), ("dev", dev_data), ("test", test_data)]:
        output_file = output_dir / f"{split_name}.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in split_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Saved {len(split_data)} samples to {output_file}")

    print("\n✓ CodeQA dataset prepared successfully!")
    return output_dir

if __name__ == "__main__":
    prepare_codeqa()
