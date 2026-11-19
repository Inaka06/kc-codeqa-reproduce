import torch

MAX_NEW_TOKENS = 32  # batas panjang jawaban (training & evaluasi harus sama)


def build_prompt_ids(tokenizer, code, question, max_prompt_len):
    """Prompt selalu diakhiri '### Answer\\n'. Kalau kepanjangan, yang dipotong HANYA kodenya."""
    head = tokenizer("### Code\n", add_special_tokens=False)["input_ids"]
    tail = tokenizer(f"\n### Question\n{question}\n### Answer\n", add_special_tokens=False)["input_ids"]
    code_ids = tokenizer(code, add_special_tokens=False)["input_ids"]
    budget = max(max_prompt_len - len(head) - len(tail), 0)
    ids = head + code_ids[:budget] + tail
    return ids[-max_prompt_len:]  # pengaman kalau pertanyaan sendiri kepanjangan


@torch.no_grad()
def generate_answers(model, tokenizer, samples, max_length=512, batch_size=16, num_beams=1):
    was_training = model.training
    model.eval()
    pad = tokenizer.pad_token_id
    preds = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        ids = [build_prompt_ids(tokenizer, s["code"], s["question"], max_length - MAX_NEW_TOKENS) for s in batch]
        L = max(len(x) for x in ids)
        input_ids = torch.tensor([[pad] * (L - len(x)) + x for x in ids]).to(model.device)
        attn = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in ids]).to(model.device)
        out = model.generate(
            input_ids=input_ids, attention_mask=attn,
            max_new_tokens=MAX_NEW_TOKENS, num_beams=num_beams, do_sample=False,
            use_cache=True, pad_token_id=pad, eos_token_id=tokenizer.eos_token_id,
        )
        for o in out:
            text = tokenizer.decode(o[L:], skip_special_tokens=True).strip()
            preds.append(text.split("\n")[0].strip())  # jawaban 1 baris; buang sisa "### Code ..." kalau model ngelantur
    if was_training:
        model.train()
    return preds