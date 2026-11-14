import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed
from transformers.modeling_outputs import BaseModelOutput

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import CodeQADataset, FiDModel  # noqa: E402
from metrics import compute_all, exact_match, f1_score  # noqa: E402
from prepare_codesearchnet import canon_hash  # noqa: E402


def load_split(args, split, tokenizer):
    data_file = Path(args.data_dir) / f"{split}.jsonl"
    if not data_file.exists():
        sys.exit(f"[ERROR] File data tidak ditemukan: {data_file}")
    retrieved_file = None
    if args.mode != "a0":
        retrieved_file = Path(args.retrieved_dir) / f"retrieved_{split}.jsonl"
        if not retrieved_file.exists():
            sys.exit(f"[ERROR] {retrieved_file} tidak ada. Mode {args.mode} butuh hasil retrieval "
                     f"(jalankan retrieve.py untuk split '{split}').")
    ds = CodeQADataset(
        data_file=str(data_file),
        retrieved_file=str(retrieved_file) if retrieved_file else None,
        tokenizer=tokenizer,
        max_input_length=args.max_input_length,
        max_output_length=args.max_new_tokens,
        mode=args.mode,
        k=args.k,
        max_passage_length=args.max_passage_length,
        passage_content=args.passage_content,
        allow_empty=args.allow_empty_retrieval,
    )
    if args.limit:
        ds.data = ds.data[: args.limit]
    return ds


def retrieval_report(ds, k, allow_empty):
    """Cek kualitas hasil retrieval: kosong, kurang dari k, dan duplikat persis dengan kode query."""
    n = len(ds)
    got = [ds.retrieved.get(i, []) for i in range(n)]
    empty = sum(1 for r in got if len(r) == 0)
    short = sum(1 for r in got if len(r) < k)
    same_code = sum(
        1 for i, r in enumerate(got)
        if any(canon_hash(x.get("code", "")) == canon_hash(ds.data[i].get("code_original") or ds.data[i]["code"])
               for x in r[:k])
    )
    rep = {"n": n, "empty": empty, "fewer_than_k": short, "exact_same_code_in_topk": same_code}
    print(f"  Retrieval: kosong={empty}/{n}, <k={short}/{n}, kode identik dgn query={same_code}/{n}")
    if empty / n > 0.01 and not allow_empty:
        sys.exit("[ERROR] >1% sampel tidak punya hasil retrieval. Kemungkinan bug di retrieve.py "
                 "(mis. tipe doc_ids tidak cocok) atau file retrieved salah. Perbaiki dulu, atau "
                 "pakai --allow_empty_retrieval jika memang disengaja.")
    if same_code > 0:
        print(f"  [WARN] {same_code} sampel memiliki kode identik di top-{k}: indikasi LEAKAGE "
              f"database retrieval. Hasil tidak valid sampai ini dibersihkan.")
    return rep


@torch.no_grad()
def generate_batch(base_model, fid, batch, args, device, use_bf16):
    gen_kwargs = dict(num_beams=args.num_beams, max_new_tokens=args.max_new_tokens, early_stopping=True, use_cache=True)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
        if args.mode == "a2":
            ids = batch["passage_input_ids"].to(device)
            mask = batch["passage_attention_mask"].to(device)
            hidden, attn = fid(ids, mask)  # tanpa labels -> (hidden_states, attention_mask)
            return base_model.generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=hidden),
                attention_mask=attn, **gen_kwargs)
        return base_model.generate(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device), **gen_kwargs)


def run_split(args, split, tokenizer, base_model, fid, device, use_bf16, results_dir):
    print(f"\n=== Evaluasi {args.exp_name} pada {split} ===")
    ds = load_split(args, split, tokenizer)
    n = len(ds)
    print(f"  Jumlah sampel: {n}")
    rep = retrieval_report(ds, args.k, args.allow_empty_retrieval) if args.mode != "a0" else None

    preds = []
    for start in tqdm(range(0, n, args.batch_size), desc=f"Generate {split}"):
        end = min(start + args.batch_size, n)
        batch = ds.collate_fn([ds[i] for i in range(start, end)])
        out = generate_batch(base_model, fid, batch, args, device, use_bf16)
        preds.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
    preds = [p.strip() for p in preds]
    refs = [ds.data[i]["answer"] for i in range(n)]

    scores = compute_all(preds, refs)
    scores["n"] = n
    if rep is not None:
        scores["retrieval"] = rep

    pred_file = results_dir / f"predictions_{args.exp_name}_{split}.jsonl"
    with open(pred_file, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({
                "idx": i, "question": ds.data[i]["question"], "code": ds.data[i]["code"],
                "reference": refs[i], "prediction": preds[i],
                "em": exact_match(preds[i], refs[i]), "f1": f1_score(preds[i], refs[i]),
            }, ensure_ascii=False) + "\n")
    print(f"  Prediksi disimpan: {pred_file}")
    return scores


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = args.bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
    print(f"Device: {device} | bf16: {use_bf16}")

    if args.limit:
        args.exp_name += "_SMOKE"
        print(f"[WARN] --limit {args.limit} aktif: hanya untuk cek pipeline. "
              f"Hasil diberi label _SMOKE dan JANGAN dilaporkan.")

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        sys.exit(f"[ERROR] Folder model tidak ditemukan: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device).eval()
    fid = FiDModel(base_model) if args.mode == "a2" else None

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = results_dir / f"metrics_{args.exp_name}.json"
    record = {"splits": {}}
    if metrics_file.exists():
        record = json.load(open(metrics_file, encoding="utf-8"))
    record.update({
        "exp": args.exp_name, "mode": args.mode, "model_dir": str(model_dir),
        "config": {k: getattr(args, k) for k in
                   ["k", "num_beams", "max_new_tokens", "max_input_length", "max_passage_length", "passage_content", "bf16", "seed"]},
    })

    for split in args.splits:
        record["splits"][split] = run_split(args, split, tokenizer, base_model, fid, device, use_bf16, results_dir)
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

    print(f"\n{'split':<6} {'BLEU':>7} {'ROUGE-L':>8} {'METEOR':>7} {'EM':>6} {'F1':>6}")
    for split, s in record["splits"].items():
        met = f"{s['meteor']:.2f}" if s.get("meteor") is not None else "  n/a"
        print(f"{split:<6} {s['bleu']:>7.2f} {s['rouge_l']:>8.2f} {met:>7} {s['em']:>6.2f} {s['f1']:>6.2f}")
    print(f"\nMetrik disimpan: {metrics_file}")
    print("Pilih model berdasarkan DEV. Jalankan TEST sekali saja untuk model final.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exp_name", required=True)
    p.add_argument("--mode", choices=["a0", "a1", "a2"], required=True)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--splits", nargs="+", default=["dev", "test"], choices=["dev", "test", "train"])
    p.add_argument("--data_dir", default="data/codeqa_processed")
    p.add_argument("--retrieved_dir", default="data")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--max_input_length", type=int, default=512)
    p.add_argument("--max_passage_length", type=int, default=256)
    p.add_argument("--passage_content", choices=["both", "summary", "code"], default="both")
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--num_beams", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--bf16", action="store_true", help="autocast bf16 saat generate (RTX 3060 mendukung)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow_empty_retrieval", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="HANYA untuk smoke test pipeline")
    main(p.parse_args())