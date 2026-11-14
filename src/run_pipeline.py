import argparse
import os
import subprocess
import sys
from pathlib import Path

PY = sys.executable
CQ = "data/codeqa_processed"


def gen_cmd(a, mode, out, k=None, bs=4, accum=8, smoke=False):
    cmd = ["src/train.py", "--mode", mode, "--output_dir", out, "--train_file", f"{CQ}/train.jsonl",
           "--dev_file", f"{CQ}/dev.jsonl", "--batch_size", str(bs), "--grad_accum", str(accum)]
    if a.profile == "fast":
        cmd += ["--lora", "--max_epochs", "6", "--patience", "2"]
        if not a.ckpt:
            cmd += ["--no-grad_ckpt"]
    if mode != "a0":
        cmd += ["--k", str(k), "--train_retrieved", "data/retrieved_train.jsonl",
                "--dev_retrieved", "data/retrieved_dev.jsonl"]
    cmd += ["--smoke"] if smoke else ["--resume"]
    return cmd


def stages(a):
    fast = a.profile == "fast"
    samples = a.retriever_samples if a.retriever_samples is not None else (100000 if fast else 200000)
    epochs = a.retriever_epochs if a.retriever_epochs is not None else (2 if fast else 3)
    csn = ["src/prepare_codesearchnet.py"] + (["--local_dir", a.csn_local_dir] if a.csn_local_dir else [])
    tr = ["src/train_retriever.py", "--output_dir", "checkpoints/retriever", "--batch_size", str(a.retriever_batch),
          "--num_epochs", str(epochs), "--resume"] + (["--max_train_samples", str(samples)] if samples else [])
    ev = ["src/evaluate.py", "--bf16", "--splits", "dev", "test"]
    bs0, bs2 = (8, 2) if fast else (4, 2)
    ac0, ac2 = (4, 16) if fast else (8, 16)
    S = [
        ("prepare_codeqa", ["src/prepare_codeqa.py"], f"{CQ}/test.jsonl"),
        ("prepare_csn", csn, "data/csn_database/database.jsonl"),
        ("train_retriever", tr, "checkpoints/retriever/query_encoder/config.json"),
        ("build_index", ["src/build_index.py", "--retriever_path", "checkpoints/retriever",
                         "--output_dir", "data/faiss_index"], "data/faiss_index/index.faiss"),
    ]
    for sp in ("train", "dev", "test"):
        S.append((f"retrieve_{sp}", ["src/retrieve.py", "--retriever_path", "checkpoints/retriever",
                                     "--index_path", "data/faiss_index", "--input_file", f"{CQ}/{sp}.jsonl",
                                     "--output_file", f"data/retrieved_{sp}.jsonl", "--top_k", "10"],
                  f"data/retrieved_{sp}.jsonl"))
    S += [
        ("train_a0", gen_cmd(a, "a0", "checkpoints/a0", bs=bs0, accum=ac0), "checkpoints/a0/config.json"),
        ("train_a2", gen_cmd(a, "a2", "checkpoints/a2", k=5, bs=bs2, accum=ac2), "checkpoints/a2/config.json"),
        ("train_a1", gen_cmd(a, "a1", "checkpoints/a1", k=3, bs=4, accum=8), "checkpoints/a1/config.json"),
        ("eval_a0", ev + ["--exp_name", "a0", "--mode", "a0", "--model_dir", "checkpoints/a0"], "results/metrics_a0.json"),
        ("eval_a2", ev + ["--exp_name", "a2", "--mode", "a2", "--model_dir", "checkpoints/a2", "--k", "5"],
         "results/metrics_a2.json"),
        ("eval_a1", ev + ["--exp_name", "a1", "--mode", "a1", "--model_dir", "checkpoints/a1", "--k", "3"],
         "results/metrics_a1.json"),
    ]
    return S


def smoke_stages(a):
    """Pre-flight: cek memori/pipeline pada beberapa step. Output *_smoke atau checkpoints sementara; JANGAN dilaporkan."""
    bs0, bs2 = (8, 2) if a.profile == "fast" else (4, 2)
    ac0, ac2 = (4, 16) if a.profile == "fast" else (8, 16)
    return [
        ("smoke_retriever", ["src/train_retriever.py", "--output_dir", "checkpoints/retriever_smoke", "--smoke",
                             "--batch_size", str(a.retriever_batch), "--max_train_samples", "2000"], None),
        ("smoke_a0", gen_cmd(a, "a0", "checkpoints/a0", bs=bs0, accum=ac0, smoke=True), None),
        ("smoke_a2", gen_cmd(a, "a2", "checkpoints/a2", k=5, bs=bs2, accum=ac2, smoke=True), None),
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", choices=["fast", "full"], default="fast")
    p.add_argument("--ckpt", action="store_true", help="profil fast: nyalakan gradient checkpointing (jika OOM)")
    p.add_argument("--only", default=None, help="nama tahap, atau 'smoke'")
    p.add_argument("--from_stage", default=None)
    p.add_argument("--until", default=None, help="berhenti setelah tahap ini (mis. retrieve_test)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--csn_local_dir", default=None)
    p.add_argument("--retriever_batch", type=int, default=32)
    p.add_argument("--retriever_epochs", type=int, default=None)
    p.add_argument("--retriever_samples", type=int, default=None, help="0 = semua pasangan database")
    a = p.parse_args()

    S = smoke_stages(a) if a.only == "smoke" else stages(a)
    names = [s[0] for s in S]
    if a.only and a.only != "smoke":
        S = [s for s in S if s[0] == a.only]
    if a.from_stage:
        if a.from_stage not in names:
            raise SystemExit(f"Tahap tidak dikenal: {a.from_stage}. Pilihan: {names}")
        S = S[names.index(a.from_stage):]
    if a.until:
        if a.until not in names:
            raise SystemExit(f"Tahap tidak dikenal: {a.until}. Pilihan: {names}")
        S = [s for s in S if names.index(s[0]) <= names.index(a.until)]
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONPATH": str(Path("src").resolve())}
    print(f"Profil: {a.profile}")
    for name, cmd, sentinel in S:
        if sentinel and Path(sentinel).exists() and not a.force:
            print(f"[SKIP] {name} (sudah ada: {sentinel})")
            continue
        print(f"\n===== {name} =====\n{PY} {' '.join(cmd)}")
        if a.dry_run:
            continue
        if subprocess.run([PY] + cmd, env=env).returncode != 0:
            raise SystemExit(f"[GAGAL] tahap {name}. Perbaiki lalu jalankan ulang (tahap selesai akan dilewati).")
    print("\nSelesai.")


if __name__ == "__main__":
    main()