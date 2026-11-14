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


_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SKIP = {"STRING", "_NUM", "DCNL", "DCSP"}  # penanda literal/newline yang bukan identifier asli


def ident_tokens(code_str):
    """Token identifier (termasuk keyword) setelah docstring/komentar/string/angka dibuang."""
    return [t for t in _IDENT_RE.findall(normalize_code(code_str)) if t not in _SKIP]


def squash(code_str):
    """Bentuk kanonik untuk perbandingan: identifier digabung tanpa spasi/underscore, huruf kecil."""
    return ''.join(ident_tokens(code_str)).replace('_', '').lower()


def canon_hash(code_str):
    return hashlib.md5(squash(code_str).encode('utf-8')).hexdigest()


def shingles(sq, n=5):
    if not sq:
        return set()
    if len(sq) < n:
        return {sq}
    return {sq[i:i + n] for i in range(len(sq) - n + 1)}


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
    m.update_batch([x.encode('utf-8') for x in sh])
    return m


def build_leak_index(codeqa_codes, num_perm=128, lsh_threshold=0.7):
    """Index kode CodeQA (unik menurut bentuk squash). LSH memakai ambang lebih longgar untuk kandidat;
    keputusan akhir memakai estimasi Jaccard MinHash >= near_dup_threshold."""
    from datasketch import MinHashLSH
    uniq = {}
    for c in codeqa_codes:
        sq = squash(c)
        if sq:
            uniq.setdefault(hashlib.md5(sq.encode('utf-8')).hexdigest(), sq)
    lsh = MinHashLSH(threshold=lsh_threshold, num_perm=num_perm)
    mh = {}
    for i, sq in enumerate(uniq.values()):
        m = _minhash(shingles(sq), num_perm)
        mh[str(i)] = m
        lsh.insert(str(i), m)
    print(f"Indeks leakage: {len(uniq)} kode CodeQA unik")
    return set(uniq.keys()), lsh, mh


# ----------------------------------------------------------------------------- database
def prepare_csn_database(records, codeqa_codes, output_dir="data/csn_database",
                         near_dup_threshold=0.8, num_perm=128, max_functions=0):
    print("\n=== Menyiapkan database CodeSearchNet ===\n")
    leak_hashes, lsh, mh = build_leak_index(codeqa_codes, num_perm=num_perm,
                                            lsh_threshold=max(near_dup_threshold - 0.1, 0.5))

    stats = dict(total=0, skipped_empty=0, removed_internal_dup=0, removed_exact_codeqa=0,
                 removed_near_codeqa=0, final=0, near_dup_threshold=near_dup_threshold)
    seen, database = set(), []

    for code, doc in records:
        stats['total'] += 1
        if stats['total'] % 20000 == 0:
            print(f"  {stats['total']} diproses, {len(database)} lolos")
        if not code or not doc:
            stats['skipped_empty'] += 1
            continue
        summary = extract_first_sentence(doc, 64)
        toks = ident_tokens(code)
        sq = ''.join(toks).replace('_', '').lower()
        if not summary or not sq:
            stats['skipped_empty'] += 1
            continue
        h = hashlib.md5(sq.encode('utf-8')).hexdigest()
        if h in seen:
            stats['removed_internal_dup'] += 1
            continue
        seen.add(h)
        if h in leak_hashes:
            stats['removed_exact_codeqa'] += 1
            continue
        if len(sq) >= 20:
            m = _minhash(shingles(sq), num_perm)
            if any(m.jaccard(mh[key]) >= near_dup_threshold for key in lsh.query(m)):
                stats['removed_near_codeqa'] += 1
                continue
        database.append({'code': ' '.join(toks), 'summary': summary, 'code_hash': h})
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
            for l in f:
                if l.strip():
                    rec = json.loads(l)
                    codes.append(rec['code'])
                    if rec.get('code_original'):
                        codes.append(rec['code_original'])
    print(f"Memuat {len(codes)} kode CodeQA (train+dev+test)")
    return codes


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--local_dir", default=None, help="Folder berisi *.jsonl.gz CSN python train")
    p.add_argument("--codeqa_dir", default="data/codeqa_processed")
    p.add_argument("--output_dir", default="data/csn_database")
    p.add_argument("--max_functions", type=int, default=0, help="0 = semua")
    p.add_argument("--near_threshold", type=float, default=0.8, help="ambang Jaccard shingle karakter")
    a = p.parse_args()
    prepare_csn_database(iter_csn_records(a.local_dir), load_codeqa_codes(a.codeqa_dir),
                         output_dir=a.output_dir, max_functions=a.max_functions,
                         near_dup_threshold=a.near_threshold)