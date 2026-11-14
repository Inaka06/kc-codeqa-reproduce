import argparse
import json
import sys
from pathlib import Path

DRIVE_URL = "https://drive.google.com/drive/folders/1i04sJNUHwMuDfMV2UfWeQG-Uv8MRw_qh"
EXPECTED = {"train": 56085, "dev": 7000, "test": 7000}
SPLITS = ("train", "dev", "test")


def fail(msg):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def split_file(lang_dir, split, ext):
    for p in (lang_dir / split / f"{split}.{ext}", lang_dir / f"{split}.{ext}"):
        if p.exists():
            return p
    return None


def find_lang_dir(raw_dir):
    raw = Path(raw_dir)
    if not raw.exists():
        return None
    cands = []
    for p in raw.rglob("train.code"):
        lang_dir = p.parent.parent if p.parent.name == "train" else p.parent
        if all(split_file(lang_dir, s, "code") for s in SPLITS):
            cands.append(lang_dir)
    cands = sorted(set(cands))
    py = [c for c in cands if "python" in str(c).lower()]
    if len(py) == 1:
        return py[0]
    if len(cands) == 1 and not py:
        return cands[0]
    if len(py) > 1 or len(cands) > 1:
        print("Ditemukan beberapa kandidat, tentukan dengan --lang_dir:", file=sys.stderr)
        for c in (py or cands):
            print("  ", c, file=sys.stderr)
        sys.exit(1)
    return None


def try_download(raw_dir):
    try:
        import gdown
    except ImportError:
        print("gdown belum terpasang: pip install gdown", file=sys.stderr)
        return False
    try:
        Path(raw_dir).mkdir(parents=True, exist_ok=True)
        gdown.download_folder(url=DRIVE_URL, output=str(raw_dir), quiet=False, use_cookies=False)
        return True
    except Exception as e:
        print(f"Unduhan gdown gagal: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
        return False


def read_lines(path):
    with open(path, encoding="utf-8") as f:  # iterasi baris, sama seperti run.py di repo CodeQA
        return [l.rstrip("\n").rstrip("\r").strip() for l in f]


def load_split(lang_dir, split):
    files = {ext: split_file(lang_dir, split, ext) for ext in ("code", "question", "answer", "code.original")}
    for ext in ("code", "question", "answer"):
        if files[ext] is None:
            fail(f"Berkas {split}.{ext} tidak ditemukan di {lang_dir}")
    cols = {ext: read_lines(p) for ext, p in files.items() if p is not None}
    n = len(cols["code"])
    for ext, c in cols.items():
        if len(c) != n:
            fail(f"Jumlah baris tidak sama pada split {split}: code={n}, {ext}={len(c)}")
    rows = []
    for i in range(n):
        r = {"question": cols["question"][i], "code": cols["code"][i], "answer": cols["answer"][i]}
        if "code.original" in cols:
            r["code_original"] = cols["code.original"][i]
        rows.append(r)
    return rows


def main(a):
    lang_dir = Path(a.lang_dir) if a.lang_dir else find_lang_dir(a.raw_dir)
    if lang_dir is None and a.download:
        if try_download(a.raw_dir):
            lang_dir = find_lang_dir(a.raw_dir)
    if lang_dir is None:
        fail("Data CodeQA mentah tidak ditemukan.\n"
             f"  Repo GitHub CodeQA hanya berisi contoh. Unduh data lengkap dari Google Drive:\n  {DRIVE_URL}\n"
             f"  lalu unzip ke {a.raw_dir}/ sehingga ada {a.raw_dir}/python/train/train.code dst.\n"
             "  (atau gunakan --lang_dir <folder python>, atau --download untuk mencoba gdown)")
    print(f"Memakai data dari: {lang_dir}")
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = {s: load_split(lang_dir, s) for s in SPLITS}
    for s, rows in data.items():
        flag = "" if len(rows) == EXPECTED[s] else f"  <-- BERBEDA dari paper ({EXPECTED[s]})"
        print(f"{s}: {len(rows)} sampel{flag}")
    tr = data["train"]
    avg = lambda key: sum(len(r[key].split()) for r in tr) / max(len(tr), 1)
    print(f"Rata-rata token (train): kode {avg('code'):.1f} (paper ~49) | pertanyaan {avg('question'):.1f} (~8) | "
          f"jawaban {avg('answer'):.1f} (~4) | kode unik {len({r['code'] for r in tr})} (paper ~34.641)")
    empty = sum(1 for rows in data.values() for r in rows if not (r["question"] and r["answer"] and r["code"]))
    if empty:
        print(f"[WARN] {empty} sampel punya kolom kosong")
    print("code_original tersedia:", "code_original" in tr[0])
    for s, rows in data.items():
        with open(out / f"{s}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] Disimpan di {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", default="data/codeqa_raw")
    p.add_argument("--lang_dir", default=None, help="folder python yang berisi train/ dev/ test/")
    p.add_argument("--output_dir", default="data/codeqa_processed")
    p.add_argument("--download", action="store_true", help="coba unduh dari Google Drive dengan gdown (best-effort)")
    main(p.parse_args())