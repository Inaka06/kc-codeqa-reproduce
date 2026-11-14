import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path


# ----------------------------------------------------------------------------- normalisasi
def normalize_code(code_str):
    """Normalisasi ala CodeQA: tanpa docstring/komentar, string -> STRING, angka -> _NUM."""
    if not code_str:
        return ""
    code_str = re.sub(r'"""[\s\S]*?"""', '', code_str)
    code_str = re.sub(r"'''[\s\S]*?'''", '', code_str)
    code_str = re.sub(r'#.*$', '', code_str, flags=re.MULTILINE)
    code_str = re.sub(r'"[^"\n]*"', 'STRING', code_str)
    code_str = re.sub(r"'[^'\n]*'", 'STRING', code_str)
    code_str = re.sub(r'\b\d+\.?\d*\b', '_NUM', code_str)
    return '\n'.join(line.strip() for line in code_str.split('\n') if line.strip())


_TOKEN_RE = re.compile(r"[A-Za-z_]\w*|\d+|\S")
_SKIP = {"DCNL", "DCSP"}  # penanda newline/indent pada korpus Python asli (jika ada)


def canon_tokens(code_str):
    """Token kanonik, tidak peka spasi/indentasi/format, dipakai HANYA untuk perbandingan."""
    return [t for t in _TOKEN_RE.findall(normalize_code(code_str)) if t not in _SKIP]


def canon_hash(code_str):
    return hashlib.md5(' '.join(canon_tokens(code_str)).encode('utf-8')).hexdigest()


def shingles(tokens, n=3):
    if len(tokens) < n:
        return {' '.join(tokens)}
    return {' '.join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def jaccard(a, b):
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def extract_first_sentence(docstring, max_tokens=64):
    if not docstring:
        return ""
    para = re.split(r'\n\s*\n', docstring.strip())[0]
    para = ' '.join(para.split())
    first = re.split(r'(?<=[.!?])\s+', para)[0]
    return ' '.join(first.split()[:max_tokens]).strip()


# ----------------------------------------------------------------------------- sumber data
def iter_csn_records(local_dir=None):
    """Hasilkan (code, docstring) dari CSN Python train."""
    if local_dir:
        files = sorted(Path(local_dir).rglob('*.jsonl.gz'))
        if not files:
            raise SystemExit(f"[ERROR] Tidak ada *.jsonl.gz di {local_dir}")
        for f in files:
            with gzip.open(f, 'rt', encoding='utf-8') as fh:
                for line in fh:
                    if line.strip():
                        d = json.loads(line)
                        yield (d.get('code') or d.get('func_code_string') or '',
                               d.get('docstring') or d.get('func_documentation_string') or '')
        return
    from datasets import load_dataset
    ds = None
    for name in ("code_search_net", "code-search-net/code_search_net"):
        try:
            print(f"Memuat {name} (python, train) dari HuggingFace...")
            ds = load_dataset(name, "python", split="train")
            break
        except Exception as e:
            print(f"  gagal: {type(e).__name__}: {str(e)[:200]}")
    if ds is None:
        raise SystemExit("[ERROR] Tidak bisa memuat CSN dari HuggingFace. Lihat petunjuk --local_dir di "
                         "bagian atas file ini.")
    print(f"CSN Python train: {len(ds)} fungsi")
    for item in ds:
        yield item.get('func_code_string', ''), item.get('func_documentation_string', '')


# ----------------------------------------------------------------------------- leakage index
def _minhash(sh, num_perm):
    from datasketch import MinHash
    m = MinHash(num_perm=num_perm)
    m.update_batch([s.encode('utf-8') for s in sh])
    return m


def build_leak_index(codeqa_codes, num_perm=128, lsh_threshold=0.8):
    """Index kode CodeQA (unik, kanonik). LSH memakai ambang 0.8 untuk kandidat, lalu diverifikasi 0.9."""
    from datasketch import MinHashLSH
    uniq = {}
    for c in codeqa_codes:
        toks = canon_tokens(c)
        if toks:
            uniq.setdefault(hashlib.md5(' '.join(toks).encode('utf-8')).hexdigest(), toks)
    lsh = MinHashLSH(threshold=lsh_threshold, num_perm=num_perm)
    sh_by_key = {}
    for i, toks in enumerate(uniq.values()):
        s = shingles(toks)
        key = str(i)
        sh_by_key[key] = s
        lsh.insert(key, _minhash(s, num_perm))
    print(f"Indeks leakage: {len(uniq)} kode CodeQA unik")
    return set(uniq.keys()), lsh, sh_by_key


# ----------------------------------------------------------------------------- database
def prepare_csn_database(records, codeqa_codes, output_dir="data/csn_database",
                         near_dup_threshold=0.9, num_perm=128, max_functions=0):
    print("\n=== Menyiapkan database CodeSearchNet ===\n")
    leak_hashes, lsh, sh_by_key = build_leak_index(codeqa_codes, num_perm=num_perm)

    stats = dict(total=0, skipped_empty=0, removed_internal_dup=0, removed_exact_codeqa=0,
                 removed_near_codeqa=0, final=0)
    seen, database = set(), []

    for code, doc in records:
        stats['total'] += 1
        if stats['total'] % 20000 == 0:
            print(f"  {stats['total']} diproses, {len(database)} lolos")
        if not code or not doc:
            stats['skipped_empty'] += 1
            continue
        summary = extract_first_sentence(doc, 64)
        norm = normalize_code(code)
        toks = canon_tokens(code)
        if not summary or not norm or not toks:
            stats['skipped_empty'] += 1
            continue
        h = hashlib.md5(' '.join(toks).encode('utf-8')).hexdigest()
        if h in seen:
            stats['removed_internal_dup'] += 1
            continue
        seen.add(h)
        if h in leak_hashes:
            stats['removed_exact_codeqa'] += 1
            continue
        sh = shingles(toks)
        near = False
        for key in lsh.query(_minhash(sh, num_perm)):
            if jaccard(sh, sh_by_key[key]) >= near_dup_threshold:
                near = True
                break
        if near:
            stats['removed_near_codeqa'] += 1
            continue
        database.append({'code': norm, 'summary': summary, 'code_hash': h})
        if max_functions and len(database) >= max_functions:
            break

    stats['final'] = len(database)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "database.jsonl", 'w', encoding='utf-8') as f:
        for e in database:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')
    with open(out / "stats.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print("\nStatistik pembersihan:", json.dumps(stats, indent=2))
    print(f"[OK] Database disimpan: {out / 'database.jsonl'}")
    return database, stats


def load_codeqa_codes(codeqa_dir="data/codeqa_processed"):
    d = Path(codeqa_dir)
    if not d.exists():
        raise SystemExit("[ERROR] Jalankan prepare_codeqa.py dulu (leakage check butuh kode CodeQA).")
    codes = []
    for split in ('train', 'dev', 'test'):
        with open(d / f"{split}.jsonl", 'r', encoding='utf-8') as f:
            codes.extend(json.loads(l)['code'] for l in f if l.strip())
    print(f"Memuat {len(codes)} kode CodeQA (train+dev+test)")
    return codes


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--local_dir", default=None, help="Folder berisi *.jsonl.gz CSN python train")
    p.add_argument("--codeqa_dir", default="data/codeqa_processed")
    p.add_argument("--output_dir", default="data/csn_database")
    p.add_argument("--max_functions", type=int, default=0, help="0 = semua")
    a = p.parse_args()
    prepare_csn_database(iter_csn_records(a.local_dir), load_codeqa_codes(a.codeqa_dir),
                         output_dir=a.output_dir, max_functions=a.max_functions)