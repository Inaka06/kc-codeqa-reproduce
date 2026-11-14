import argparse
import functools
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


def load_pairs(database_file):
    pairs = []
    with open(database_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                it = json.loads(line)
                if it.get('code') and it.get('summary'):
                    pairs.append({'code': it['code'], 'summary': it['summary']})
    return pairs


class BiEncoder(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.query_encoder = AutoModel.from_pretrained(model_name)
        self.doc_encoder = AutoModel.from_pretrained(model_name)

    def encode_query(self, ids, mask):
        return self.query_encoder(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0, :]

    def encode_doc(self, ids, mask):
        return self.doc_encoder(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0, :]


def collate_fn(batch, tokenizer, max_code_len, max_summary_len):
    codes = [b['code'] for b in batch]
    sums = [b['summary'] for b in batch]
    q = tokenizer(codes, max_length=max_code_len, padding=True, truncation=True, return_tensors='pt')
    d = tokenizer(sums, max_length=max_summary_len, padding=True, truncation=True, return_tensors='pt')
    uniq = {}
    groups = torch.tensor([uniq.setdefault(s, len(uniq)) for s in sums])
    return {'q_ids': q['input_ids'], 'q_mask': q['attention_mask'],
            'd_ids': d['input_ids'], 'd_mask': d['attention_mask'], 'groups': groups}


def dpr_loss(q_emb, d_emb, groups):
    scores = torch.matmul(q_emb.float(), d_emb.float().T)
    same = (groups[:, None] == groups[None, :]) & ~torch.eye(len(groups), dtype=torch.bool, device=groups.device)
    scores = scores.masked_fill(same, -1e4)
    labels = torch.arange(scores.size(0), device=scores.device)
    return torch.nn.functional.cross_entropy(scores, labels)


@torch.no_grad()
def evaluate_retrieval(model, pairs, tokenizer, device, args, use_bf16):
    model.eval()
    qs, ds = [], []
    for i in range(0, len(pairs), 128):
        b = collate_fn(pairs[i:i + 128], tokenizer, args.max_code_len, args.max_summary_len)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            qs.append(model.encode_query(b['q_ids'].to(device), b['q_mask'].to(device)).float().cpu())
            ds.append(model.encode_doc(b['d_ids'].to(device), b['d_mask'].to(device)).float().cpu())
    scores = torch.cat(qs) @ torch.cat(ds).T
    diag = scores.diag().unsqueeze(1)
    ranks = (scores > diag).sum(1) + 1
    n = len(pairs)
    res = {'mrr': float((1.0 / ranks.float()).mean())}
    for k in (1, 5, 10):
        res[f'recall@{k}'] = float((ranks <= k).float().mean())
    res['n_candidates'] = n
    return res


def save_models(model, tokenizer, out):
    out.mkdir(parents=True, exist_ok=True)
    model.query_encoder.save_pretrained(out / "query_encoder")
    model.doc_encoder.save_pretrained(out / "doc_encoder")
    tokenizer.save_pretrained(out)


def main(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_bf16 = device.type == 'cuda' and torch.cuda.is_bf16_supported() and not args.no_bf16
    print(f"Device: {device} | bf16: {use_bf16}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = BiEncoder(args.model_name).to(device)
    if args.grad_ckpt:
        model.query_encoder.gradient_checkpointing_enable()
        model.doc_encoder.gradient_checkpointing_enable()

    pairs = load_pairs(args.database_file)
    print(f"Memuat {len(pairs)} pasangan (kode, ringkasan)")
    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    val_pairs, train_pairs = pairs[:args.val_size], pairs[args.val_size:]
    if args.max_train_samples and args.max_train_samples < len(train_pairs):
        train_pairs = train_pairs[:args.max_train_samples]
    print(f"Train: {len(train_pairs)} | Validasi: {len(val_pairs)}")

    loader = DataLoader(train_pairs, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        collate_fn=functools.partial(collate_fn, tokenizer=tokenizer,
                                                     max_code_len=args.max_code_len,
                                                     max_summary_len=args.max_summary_len),
                        num_workers=args.num_workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total * args.warmup_ratio), total)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "retriever_state.pt"
    start_epoch, best_mrr = 0, -1.0
    if args.resume and state_path.exists():
        st = torch.load(state_path, map_location='cpu', weights_only=False)
        model.load_state_dict(st['model']); optimizer.load_state_dict(st['optimizer'])
        scheduler.load_state_dict(st['scheduler'])
        start_epoch, best_mrr = st['epoch'] + 1, st['best_mrr']
        print(f"Resume dari epoch {start_epoch}, best MRR {best_mrr:.4f}")

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        total_loss, nb = 0.0, 0
        for step, b in enumerate(tqdm(loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}")):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                q = model.encode_query(b['q_ids'].to(device), b['q_mask'].to(device))
                d = model.encode_doc(b['d_ids'].to(device), b['d_mask'].to(device))
            loss = dpr_loss(q, d, b['groups'].to(device))
            if not torch.isfinite(loss):
                raise RuntimeError("Loss NaN/inf pada training retriever")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            total_loss += loss.item(); nb += 1
            if args.smoke and step + 1 >= 20:
                break
        val = evaluate_retrieval(model, val_pairs, tokenizer, device, args, use_bf16)
        rec = {'epoch': epoch + 1, 'train_loss': total_loss / max(nb, 1), **val}
        print(json.dumps(rec))
        with open(out / "retriever_log.jsonl", 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec) + '\n')
        if val['mrr'] > best_mrr:
            best_mrr = val['mrr']
            save_models(model, tokenizer, out)
            print(f"[OK] Checkpoint terbaik (MRR {best_mrr:.4f}) disimpan di {out}")
        tmp = str(state_path) + ".tmp"
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'epoch': epoch, 'best_mrr': best_mrr}, tmp)
        os.replace(tmp, state_path)
        if args.smoke:
            break
    if device.type == 'cuda':
        print(f"Memori GPU puncak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"\n[OK] Selesai. MRR validasi terbaik: {best_mrr:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--database_file", default="data/csn_database/database.jsonl")
    p.add_argument("--model_name", default="microsoft/graphcodebert-base")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_code_len", type=int, default=256)
    p.add_argument("--max_summary_len", type=int, default=64)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--val_size", type=int, default=2000)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--no_bf16", action="store_true")
    p.add_argument("--grad_ckpt", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true", help="Pre-flight check 20 step; JANGAN dipakai untuk hasil")
    main(p.parse_args())