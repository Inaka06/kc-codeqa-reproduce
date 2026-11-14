import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from transformers.modeling_outputs import BaseModelOutput

CSEP, NSEP = "[csep]", "[nsep]"


class CodeQADataset(Dataset):
    def __init__(self, data_file, retrieved_file=None, tokenizer=None, max_input_length=512,
                 max_output_length=32, mode='a0', k=5, max_passage_length=256,
                 passage_content='both', allow_empty=False):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.mode = mode
        self.k = k
        self.max_passage_length = max_passage_length
        self.passage_content = passage_content  # both | summary | code  (ablasi A4)

        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f if line.strip()]

        self.retrieved = {}
        if mode != 'a0':
            if not retrieved_file:
                raise ValueError(f"Mode {mode} butuh file retrieval (--train_retrieved/--dev_retrieved).")
            if not Path(retrieved_file).exists():
                raise FileNotFoundError(f"File retrieval tidak ditemukan: {retrieved_file}")
            with open(retrieved_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        self.retrieved[item['idx']] = item.get('retrieved', [])
            n = len(self.data)
            empty = sum(1 for i in range(n) if not self.retrieved.get(i))
            print(f"  Retrieval {Path(retrieved_file).name}: {len(self.retrieved)} baris, "
                  f"{empty}/{n} sampel tanpa hasil")
            if empty > 0.01 * n and not allow_empty:
                raise RuntimeError(f"{empty}/{n} sampel tanpa hasil retrieval (>1%). Periksa retrieve.py "
                                   f"atau gunakan --allow_empty_retrieval.")

    def __len__(self):
        return len(self.data)

    def _passage(self, q, code, r):
        parts = [f"question: {q}"]
        if self.passage_content in ('both', 'summary'):
            parts.append(f"summary: {r.get('summary', '')}")
        parts.append(f"code: {code}")
        if self.passage_content in ('both', 'code'):
            parts.append(f"context: {r.get('code', '')}")
        return ' '.join(parts)

    def __getitem__(self, idx):
        item = self.data[idx]
        q, code, answer = item['question'], item['code'], item['answer']
        if self.mode == 'a0':
            return {'input_text': f"question: {q} context: {code}", 'target_text': answer, 'idx': idx}
        retrieved = self.retrieved.get(idx, [])[:self.k]
        if self.mode == 'a1':
            parts = [q, CSEP, code]
            for r in retrieved:
                parts.extend([CSEP, r.get('summary', ''), NSEP, r.get('code', '')])
            return {'input_text': ' '.join(parts), 'target_text': answer, 'idx': idx}
        passages = [self._passage(q, code, r) for r in retrieved]
        while len(passages) < self.k:
            passages.append(self._passage(q, code, {}))
        return {'passages': passages, 'target_text': answer, 'idx': idx}

    def collate_fn(self, batch):
        tok = self.tokenizer
        targets = tok([b['target_text'] for b in batch], max_length=self.max_output_length,
                      padding=True, truncation=True, return_tensors='pt')
        labels = targets['input_ids']
        labels[labels == tok.pad_token_id] = -100
        if self.mode in ('a0', 'a1'):
            inp = tok([b['input_text'] for b in batch], max_length=self.max_input_length,
                      padding=True, truncation=True, return_tensors='pt')
            return {'input_ids': inp['input_ids'], 'attention_mask': inp['attention_mask'], 'labels': labels}
        flat = [p for b in batch for p in b['passages']]
        enc = tok(flat, max_length=self.max_passage_length, padding=True, truncation=True, return_tensors='pt')
        bs = len(batch)
        return {'passage_input_ids': enc['input_ids'].view(bs, self.k, -1),
                'passage_attention_mask': enc['attention_mask'].view(bs, self.k, -1),
                'labels': labels}


class FiDModel(torch.nn.Module):
    """Tiap (pertanyaan + passage) di-encode independen; decoder meng-attend ke seluruh k*L state."""

    def __init__(self, base_model):
        super().__init__()
        self.model = base_model

    def forward(self, passage_input_ids, passage_attention_mask, labels=None):
        b, k, L = passage_input_ids.shape
        enc = self.model.encoder(input_ids=passage_input_ids.reshape(b * k, L),
                                 attention_mask=passage_attention_mask.reshape(b * k, L), return_dict=True)
        hidden = enc.last_hidden_state.reshape(b, k * L, -1)
        attn = passage_attention_mask.reshape(b, k * L)
        if labels is None:
            return hidden, attn
        out = self.model(encoder_outputs=BaseModelOutput(last_hidden_state=hidden), attention_mask=attn,
                         labels=labels, return_dict=True)
        return out.loss


def compute_loss(model, batch, mode, device):
    if mode == 'a2':
        return model(batch['passage_input_ids'].to(device), batch['passage_attention_mask'].to(device),
                     labels=batch['labels'].to(device))
    return model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device),
                 labels=batch['labels'].to(device)).loss


@torch.no_grad()
def generate_answers(model, batch, mode, device, num_beams=1, max_new_tokens=32):
    if mode == 'a2':
        hidden, attn = model(batch['passage_input_ids'].to(device), batch['passage_attention_mask'].to(device))
        return model.model.generate(encoder_outputs=BaseModelOutput(last_hidden_state=hidden),
                                    attention_mask=attn, num_beams=num_beams,
                                    max_new_tokens=max_new_tokens, use_cache=True)
    return model.generate(input_ids=batch['input_ids'].to(device),
                          attention_mask=batch['attention_mask'].to(device), num_beams=num_beams,
                          max_new_tokens=max_new_tokens, use_cache=True)


@torch.no_grad()
def evaluate_dev(model, dev_ds, tokenizer, args, device, use_bf16):
    import sacrebleu
    from metrics import f1_score
    model.eval()
    n = len(dev_ds)
    losses, preds = [], []
    for s in range(0, n, args.eval_batch_size):
        batch = dev_ds.collate_fn([dev_ds[i] for i in range(s, min(s + args.eval_batch_size, n))])
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            losses.append(compute_loss(model, batch, args.mode, device).item())
            out = generate_answers(model, batch, args.mode, device, args.eval_num_beams, args.max_output_length)
        preds.extend(p.strip() for p in tokenizer.batch_decode(out, skip_special_tokens=True))
    refs = [dev_ds.data[i]['answer'] for i in range(n)]
    return {'dev_loss': float(np.mean(losses)),
            'dev_bleu': sacrebleu.corpus_bleu(preds, [refs]).score,
            'dev_f1': 100 * float(np.mean([f1_score(p, r) for p, r in zip(preds, refs)])),
            'dev_n': n}


def base_of(model, mode):
    return model.model if mode == 'a2' else model


LORA_TARGETS = ["q", "k", "v", "o", "wi", "wo"]  # nama modul atensi dan FFN pada T5/CodeT5


def make_lora_config(args):
    from peft import LoraConfig
    return LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                      target_modules=LORA_TARGETS, bias="none")


def trainable_state(core):
    return {n: p.detach().cpu().clone() for n, p in core.named_parameters() if p.requires_grad}


def load_trainable_state(core, state):
    params = dict(core.named_parameters())
    missing = [n for n in state if n not in params]
    if missing:
        raise RuntimeError(f"Parameter LoRA tidak cocok dengan model: {missing[:3]}")
    with torch.no_grad():
        for n, t in state.items():
            params[n].copy_(t)


def finalize_lora(args, tokenizer, out_dir):
    """Merge adapter terbaik ke bobot dasar, simpan model penuh (bisa dimuat from_pretrained biasa)."""
    from peft import get_peft_model
    best = out_dir / "lora_best.pt"
    if not best.exists():
        raise SystemExit(f"[ERROR] {best} tidak ada; tidak ada yang bisa di-merge.")
    base = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    if args.mode == 'a1':
        base.resize_token_embeddings(len(tokenizer))
    pm = get_peft_model(base, make_lora_config(args))
    load_trainable_state(pm.base_model.model, torch.load(best, map_location='cpu'))
    merged = pm.merge_and_unload()
    merged.config.use_cache = True
    merged.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[OK] LoRA di-merge; model penuh disimpan di {out_dir}")


def save_best(model, tokenizer, out_dir, mode, lora=False):
    base = base_of(model, mode)
    if lora:
        torch.save(trainable_state(base), out_dir / "lora_best.pt")
        tokenizer.save_pretrained(out_dir)
        return
    base.config.use_cache = True
    base.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    base.config.use_cache = False


def main(args):
    set_seed(args.seed)
    if args.lr is None:
        args.lr = 3e-4 if args.lora else 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_bf16 = args.bf16 and device.type == 'cuda' and torch.cuda.is_bf16_supported()
    print(f"Device: {device} | bf16: {use_bf16} | mode: {args.mode}")
    if args.mode != 'a0' and not (args.train_retrieved and args.dev_retrieved):
        raise SystemExit("[ERROR] Mode a1/a2 wajib --train_retrieved dan --dev_retrieved.")

    out_dir = Path(args.output_dir + ("_smoke" if args.smoke else ""))
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    if args.mode == 'a1':
        tokenizer.add_special_tokens({'additional_special_tokens': [CSEP, NSEP]})
        base_model.resize_token_embeddings(len(tokenizer))
    if args.merge_only:
        finalize_lora(args, tokenizer, out_dir)
        return
    base_model.config.use_cache = False
    if args.grad_ckpt:
        base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    if args.lora:
        from peft import get_peft_model
        peft_model = get_peft_model(base_model, make_lora_config(args))
        base_model = peft_model.base_model.model  # T5 dengan LoRA tersuntik; dipakai di semua tempat
        if args.mode == 'a1':  # embedding [csep]/[nsep] yang baru harus ikut dilatih
            base_model.shared.weight.requires_grad = True
    trainable = [p for p in base_model.parameters() if p.requires_grad]
    print(f"Parameter dilatih: {sum(p.numel() for p in trainable) / 1e6:.1f}M dari "
          f"{sum(p.numel() for p in base_model.parameters()) / 1e6:.1f}M "
          f"({'LoRA' if args.lora else 'full fine-tuning'}) | grad checkpointing: {args.grad_ckpt}")
    model = (FiDModel(base_model) if args.mode == 'a2' else base_model).to(device)

    ds_kw = dict(tokenizer=tokenizer, max_input_length=args.max_input_length,
                 max_output_length=args.max_output_length, mode=args.mode, k=args.k,
                 max_passage_length=args.max_passage_length, passage_content=args.passage_content,
                 allow_empty=args.allow_empty_retrieval)
    print("Memuat data...")
    train_ds = CodeQADataset(args.train_file, args.train_retrieved, **ds_kw)
    dev_ds = CodeQADataset(args.dev_file, args.dev_retrieved, **ds_kw)
    n_dev = len(dev_ds.data) if args.eval_samples <= 0 else min(args.eval_samples, len(dev_ds.data))
    if args.smoke:
        train_ds.data = train_ds.data[:args.batch_size * args.grad_accum * 2]
        n_dev = min(n_dev, 32)
        args.max_epochs = 1
    dev_ds.data = dev_ds.data[:n_dev]
    print(f"Train: {len(train_ds)} | Dev (untuk seleksi checkpoint): {len(dev_ds)}")

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=train_ds.collate_fn,
                        num_workers=args.num_workers)
    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.max_epochs
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)
    print(f"Batch efektif: {args.batch_size * args.grad_accum} | step optimizer/epoch: {steps_per_epoch}")

    best_bleu, patience_counter, start_epoch = -1.0, 0, 0
    state_path = out_dir / "last_state.pt"
    if args.resume and state_path.exists():
        st = torch.load(state_path, map_location='cpu', weights_only=False)
        if args.lora:
            load_trainable_state(base_of(model, args.mode), st['model'])
        else:
            base_of(model, args.mode).load_state_dict(st['model'])
        optimizer.load_state_dict(st['optimizer']); scheduler.load_state_dict(st['scheduler'])
        best_bleu, patience_counter, start_epoch = st['best_bleu'], st['patience'], st['epoch'] + 1
        random.setstate(st['rng']['py']); np.random.set_state(st['rng']['np'])
        torch.set_rng_state(st['rng']['torch'])
        if device.type == 'cuda' and st['rng'].get('cuda') is not None:
            torch.cuda.set_rng_state_all(st['rng']['cuda'])
        print(f"Resume dari epoch {start_epoch + 1}, best BLEU {best_bleu:.2f}, patience {patience_counter}")

    for epoch in range(start_epoch, args.max_epochs):
        t0 = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        run_loss, n_opt = 0.0, 0
        pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch + 1}/{args.max_epochs}")
        for i, batch in pbar:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                loss = compute_loss(model, batch, args.mode, device)
            if not torch.isfinite(loss):
                raise RuntimeError("Loss NaN/inf. Turunkan --lr (mis. 5e-5) atau cek data/label.")
            (loss / args.grad_accum).backward()
            run_loss += loss.item()
            if (i + 1) % args.grad_accum == 0 or i + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                n_opt += 1
                if n_opt % 50 == 0:
                    pbar.set_postfix(loss=f"{run_loss / (i + 1):.4f}")
        train_loss = run_loss / len(loader)

        m = evaluate_dev(model, dev_ds, tokenizer, args, device, use_bf16)
        rec = {'epoch': epoch + 1, 'train_loss': train_loss, **m, 'minutes': (time.time() - t0) / 60}
        if device.type == 'cuda':
            rec['gpu_peak_gb'] = torch.cuda.max_memory_allocated() / 1e9
        print(json.dumps(rec))
        with open(out_dir / "train_log.jsonl", 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec) + '\n')

        if m['dev_bleu'] > best_bleu:
            best_bleu, patience_counter = m['dev_bleu'], 0
            save_best(model, tokenizer, out_dir, args.mode, lora=args.lora)
            print(f"[OK] Checkpoint terbaik (dev BLEU {best_bleu:.2f}) disimpan di {out_dir}")
        else:
            patience_counter += 1
            print(f"Tidak membaik. Patience {patience_counter}/{args.patience}")

        tmp = str(state_path) + ".tmp"
        torch.save({'model': (trainable_state(base_of(model, args.mode)) if args.lora
                              else base_of(model, args.mode).state_dict()), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'epoch': epoch, 'best_bleu': best_bleu,
                    'patience': patience_counter,
                    'rng': {'py': random.getstate(), 'np': np.random.get_state(),
                            'torch': torch.get_rng_state(),
                            'cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None}}, tmp)
        os.replace(tmp, state_path)
        if patience_counter >= args.patience:
            print("Early stopping.")
            break

    if args.lora:
        finalize_lora(args, tokenizer, out_dir)
    print(f"\n[OK] Selesai. Dev BLEU terbaik: {best_bleu:.2f}" + (" (SMOKE, jangan dilaporkan)" if args.smoke else ""))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_file", required=True)
    p.add_argument("--dev_file", required=True)
    p.add_argument("--train_retrieved", default=None)
    p.add_argument("--dev_retrieved", default=None)
    p.add_argument("--model_name", default="Salesforce/codet5-base")
    p.add_argument("--mode", choices=['a0', 'a1', 'a2'], required=True)
    p.add_argument("--passage_content", choices=['both', 'summary', 'code'], default='both')
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=8, help="batch efektif = batch_size * grad_accum")
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--eval_samples", type=int, default=2000, help="sampel dev untuk seleksi checkpoint; 0 = semua")
    p.add_argument("--eval_num_beams", type=int, default=1)
    p.add_argument("--lr", type=float, default=None, help="default 1e-4 (full) atau 3e-4 (--lora)")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_input_length", type=int, default=512)
    p.add_argument("--max_output_length", type=int, default=32)
    p.add_argument("--max_passage_length", type=int, default=256)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--grad_ckpt", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--allow_empty_retrieval", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--lora", action="store_true", help="fine-tuning ringan dengan LoRA")
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--merge_only", action="store_true", help="hanya merge lora_best.pt ke model penuh")
    p.add_argument("--smoke", action="store_true", help="Pre-flight check (1 epoch kecil); JANGAN dilaporkan")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_workers", type=int, default=0)
    main(p.parse_args())