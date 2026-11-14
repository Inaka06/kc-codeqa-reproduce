import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from prepare_codesearchnet import canon_hash


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.retriever_path)
    model = AutoModel.from_pretrained(Path(args.retriever_path) / "query_encoder").to(device).eval()

    index = faiss.read_index(str(Path(args.index_path) / "index.faiss"))
    with open(Path(args.index_path) / "doc_ids.json", 'r') as f:
        doc_ids = json.load(f)
    print(f"Index: {index.ntotal} dokumen")

    database = {}
    with open(args.database_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if line.strip():
                database[i] = json.loads(line)
    print(f"Database: {len(database)} entri")

    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(l) for l in f if l.strip()]
    print(f"Memproses {len(data)} query dari {args.input_file}")

    fetch = args.top_k + args.extra
    dropped_same = 0
    out_rows = []
    with torch.no_grad():
        for s in tqdm(range(0, len(data), args.batch_size)):
            chunk = data[s:s + args.batch_size]
            enc = tokenizer([(it.get('code_original') or it['code']) for it in chunk], max_length=args.max_len, padding=True,
                            truncation=True, return_tensors='pt')
            enc = {k: v.to(device) for k, v in enc.items()}
            q = model(**enc).last_hidden_state[:, 0, :].float().cpu().numpy()
            scores, indices = index.search(q, fetch)
            for j, it in enumerate(chunk):
                qh = canon_hash(it.get('code_original') or it['code'])
                retrieved = []
                for sc, di in zip(scores[j], indices[j]):
                    if di < 0:
                        continue
                    e = database.get(doc_ids[di])
                    if e is None or not e.get('summary'):
                        continue
                    if e.get('code_hash') == qh:
                        dropped_same += 1
                        continue
                    retrieved.append({'code': e['code'], 'summary': e['summary'], 'score': float(sc)})
                    if len(retrieved) == args.top_k:
                        break
                out_rows.append({'idx': s + j, 'question': it['question'], 'code': it['code'],
                                 'answer': it['answer'], 'retrieved': retrieved})

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    short = sum(1 for r in out_rows if len(r['retrieved']) < args.top_k)
    print(f"[OK] {len(out_rows)} baris disimpan ke {args.output_file} | "
          f"kurang dari top_k: {short} | dibuang (kode identik dengan query): {dropped_same}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--retriever_path", required=True)
    p.add_argument("--index_path", required=True)
    p.add_argument("--database_file", default="data/csn_database/database.jsonl")
    p.add_argument("--input_file", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--extra", type=int, default=5, help="kandidat ekstra untuk filter")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_len", type=int, default=256)
    main(p.parse_args())