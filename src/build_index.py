import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.retriever_path)
    model = AutoModel.from_pretrained(Path(args.retriever_path) / "doc_encoder").to(device).eval()

    docs, doc_ids = [], []
    with open(args.database_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if line.strip():
                docs.append(json.loads(line).get('summary', ''))
                doc_ids.append(idx)
    print(f"Meng-encode {len(docs)} dokumen...")

    order = np.argsort([len(d) for d in docs])
    emb = None
    with torch.no_grad():
        for i in tqdm(range(0, len(docs), args.batch_size)):
            idx = order[i:i + args.batch_size]
            enc = tokenizer([docs[j] for j in idx], max_length=args.max_len, padding=True,
                            truncation=True, return_tensors='pt')
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc).last_hidden_state[:, 0, :].float().cpu().numpy()
            if emb is None:
                emb = np.zeros((len(docs), out.shape[1]), dtype=np.float32)
            emb[idx] = out
    print(f"Embeddings: {emb.shape}")

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "index.faiss"))
    with open(out_dir / "doc_ids.json", 'w') as f:
        json.dump(doc_ids, f)
    print(f"[OK] Index ({index.ntotal} dokumen) disimpan di {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--retriever_path", required=True)
    p.add_argument("--database_file", default="data/csn_database/database.jsonl")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_len", type=int, default=64)
    main(p.parse_args())