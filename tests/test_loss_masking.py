import torch
from transformers import AutoTokenizer

def test_loss_masking():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    code = "def add(a, b): return a + b"
    question = "What does this do?"
    answer = "returns sum"

    prompt_prefix = f"### Code\n{code}\n### Question\n{question}\n### Answer\n"
    full_text = prompt_prefix + answer + tokenizer.eos_token

    prefix_ids = tokenizer(prompt_prefix, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    labels = list(full_ids)
    # Mask prompt tokens with -100
    for i in range(len(prefix_ids)):
        labels[i] = -100

    assert len(labels) == len(full_ids)
    assert all(l == -100 for l in labels[:len(prefix_ids)])
    assert all(l != -100 for l in labels[len(prefix_ids):])
    print("Loss masking unit test passed successfully!")

if __name__ == "__main__":
    test_loss_masking()
