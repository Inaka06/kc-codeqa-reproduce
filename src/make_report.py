import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KEYS = ["bleu", "rouge_l", "meteor", "em", "f1"]
LABELS = ["BLEU", "ROUGE-L", "METEOR", "EM", "F1"]
# CodeQA (Liu & Wan 2021), Tabel 8, Python. Urutan: BLEU, ROUGE-L, METEOR, EM, F1
PAPER = {
    "Seq2seq (paper)": {"dev": (29.70, 22.18, 6.36, 2.95, 23.67), "test": (30.09, 21.69, 6.38, 2.84, 23.27)},
    "Dual Encoder (paper)": {"dev": (28.57, 17.62, 4.43, 2.22, 18.78), "test": (28.90, 17.79, 4.53, 2.22, 18.97)},
    "Transformer (paper)": {"dev": (30.85, 23.96, 7.91, 3.37, 25.30), "test": (30.69, 23.26, 7.80, 3.35, 24.59)},
    "CodeBERT (paper)": {"dev": (33.84, 30.27, 11.51, 5.50, 31.57), "test": (34.86, 30.28, 12.51, 4.93, 31.56)},
}
COLORS = ["#2a6fbb", "#d9822b", "#3a9d5d", "#b8434f", "#7b5ea7", "#8c6d3f", "#3d9fa8", "#c2578f"]


def fmt(v):
    return "n/a" if v is None else f"{v:.2f}"


def load_results(rdir):
    out = {}
    for f in sorted(Path(rdir).glob("metrics_*.json")):
        rec = json.load(open(f, encoding="utf-8"))
        name = rec.get("exp", f.stem[len("metrics_"):])
        if name.endswith("_SMOKE"):
            continue
        out[name] = rec["splits"]
    order = [n for n in ("a0", "a1", "a2") if n in out]
    return {n: out[n] for n in order + [n for n in out if n not in order]}


def tables(res, rdir):
    md = ["# Tabel hasil (CodeQA Python)\n"]
    for split in ("dev", "test"):
        md.append(f"## {split.capitalize()}\n")
        md.append("| Model | " + " | ".join(LABELS) + " |\n|---|" + "---|" * len(KEYS))
        for n, v in PAPER.items():
            md.append(f"| {n} | " + " | ".join(f"{x:.2f}" for x in v[split]) + " |")
        ours = {n: s[split] for n, s in res.items() if split in s}
        best = {k: max((s[k] for s in ours.values() if s.get(k) is not None), default=None) for k in KEYS}
        csv_rows = [["model"] + LABELS]
        for n, s in ours.items():
            cells = []
            for k in KEYS:
                t = fmt(s.get(k))
                cells.append(f"**{t}**" if s.get(k) is not None and s[k] == best[k] and len(ours) > 1 else t)
            md.append(f"| {n} | " + " | ".join(cells) + " |")
            csv_rows.append([n] + [fmt(s.get(k)) for k in KEYS])
        with open(Path(rdir) / f"results_{split}.csv", "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(csv_rows)
        md.append("")
    if "a0" in res and "test" in res["a0"]:
        md.append("## Selisih terhadap A0 (test)\n\n| Model | " + " | ".join(LABELS) + " |\n|---|" + "---|" * len(KEYS))
        for n, s in res.items():
            if n != "a0" and "test" in s:
                d = [("n/a" if s["test"].get(k) is None or res["a0"]["test"].get(k) is None
                      else f"{s['test'][k] - res['a0']['test'][k]:+.2f}") for k in KEYS]
                md.append(f"| {n} | " + " | ".join(d) + " |")
        md.append("\nSelisih kecil (sekitar 1-2 poin) bisa masih dalam rentang variasi antar seed. "
                  "Jalankan lebih dari satu seed sebelum menyimpulkan.\n")
    return md


def plot_training(ckdir, fdir):
    logs = {}
    for f in sorted(Path(ckdir).glob("*/train_log.jsonl")):
        name = f.parent.name
        if name.endswith("_smoke") or name.startswith("retriever"):
            continue
        logs[name] = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    if not logs:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, (n, L) in enumerate(logs.items()):
        c, ep = COLORS[i % len(COLORS)], [r["epoch"] for r in L]
        ax[0].plot(ep, [r["train_loss"] for r in L], "-o", color=c, ms=3, label=f"{n} train")
        ax[0].plot(ep, [r["dev_loss"] for r in L], "--s", color=c, ms=3, label=f"{n} dev")
        ax[1].plot(ep, [r["dev_bleu"] for r in L], "-o", color=c, ms=3, label=n)
        b = max(L, key=lambda r: r["dev_bleu"])
        ax[1].plot(b["epoch"], b["dev_bleu"], "*", color=c, ms=11)
    ax[0].set(title="Loss per epoch", xlabel="Epoch", ylabel="Loss")
    ax[1].set(title="BLEU dev per epoch (bintang = checkpoint terpilih)", xlabel="Epoch", ylabel="BLEU (subset dev, greedy)")
    from matplotlib.ticker import MaxNLocator
    for a in ax:
        a.grid(alpha=0.3); a.legend(fontsize=7); a.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout(); fig.savefig(fdir / "training_curves.png", dpi=160); plt.close(fig)
    return logs


def plot_bars(res, fdir):
    models = {"CodeBERT (paper)": PAPER["CodeBERT (paper)"]["test"]}
    for n, s in res.items():
        if "test" in s:
            models[n] = tuple(s["test"].get(k) for k in KEYS)  # None (mis. METEOR tanpa WordNet) tidak digambar
    fig, ax = plt.subplots(figsize=(10, 4.2))
    w = 0.8 / len(models)
    for i, (n, v) in enumerate(models.items()):
        c = "#9aa0a6" if "paper" in n else COLORS[(i - 1) % len(COLORS)]
        xs = [j + i * w - 0.4 + w / 2 for j in range(len(KEYS))]
        ax.bar(xs, [y or 0 for y in v], w * 0.92, label=n, color=c)
        for x, y in zip(xs, v):
            ax.text(x, (y or 0) + 0.4, "n/a" if y is None else f"{y:.1f}", ha="center", fontsize=6.5)
    ax.set_xticks(range(len(KEYS))); ax.set_xticklabels(LABELS)
    ax.set(ylabel="Skor (test)", title="Metrik test: model kita vs CodeBERT (paper)")
    ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fdir / "test_metrics.png", dpi=160); plt.close(fig)


def plot_retriever(ckdir, fdir):
    f = Path(ckdir) / "retriever" / "retriever_log.jsonl"
    if not f.exists():
        return False
    L = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, lab in [("mrr", "MRR"), ("recall@1", "Recall@1"), ("recall@5", "Recall@5"), ("recall@10", "Recall@10")]:
        ax.plot([r["epoch"] for r in L], [r[k] for r in L], "-o", ms=3, label=lab)
    ax.set(title=f"Retriever pada validasi ({L[0].get('n_candidates', '?')} kandidat)", xlabel="Epoch", ylabel="Skor")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fdir / "retriever.png", dpi=160); plt.close(fig)
    return True


def qualitative(res, rdir, exp, data_dir, n_each=5):
    if exp is None:
        cands = {n: s["test"]["f1"] for n, s in res.items() if "test" in s}
        if not cands:
            return None
        exp = max(cands, key=cands.get)
    pf = Path(rdir) / f"predictions_{exp}_test.jsonl"
    if not pf.exists():
        return None
    rows = [json.loads(l) for l in open(pf, encoding="utf-8") if l.strip()]
    retr = {}
    rf = Path(data_dir) / "retrieved_test.jsonl"
    if rf.exists() and exp != "a0":
        for l in open(rf, encoding="utf-8"):
            if l.strip():
                d = json.loads(l); retr[d["idx"]] = d.get("retrieved", [])
    good = [r for r in rows if r["em"] == 1 or r["f1"] >= 0.8][:n_each]
    bad = [r for r in rows if r["f1"] == 0][:n_each]
    md = [f"# Contoh kualitatif ({exp}, test)\n",
          "Contoh diambil berurutan dari indeks terkecil (bukan dipilih), supaya tidak bias.\n"]
    for title, group in (("Prediksi benar (EM atau F1 >= 0,8)", good), ("Prediksi salah (F1 = 0)", bad)):
        md.append(f"## {title}\n")
        for r in group:
            md += [f"**#{r['idx']}**  \nPertanyaan: {r['question']}  \nReferensi: `{r['reference']}`  \nPrediksi: `{r['prediction']}`  ",
                   "```python\n" + r["code"][:500] + ("..." if len(r["code"]) > 500 else "") + "\n```"]
            for p in retr.get(r["idx"], [])[:3]:
                md.append(f"- retrieved: {p.get('summary', '')} (skor {p.get('score', 0):.1f})")
            md.append("")
    return md, exp


def main(a):
    rdir, fdir = Path(a.results_dir), Path(a.results_dir) / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    res = load_results(rdir)
    if not res:
        raise SystemExit(f"[ERROR] Tidak ada metrics_*.json di {rdir}. Jalankan evaluate.py dulu.")
    md = tables(res, rdir)
    sf = Path(a.data_dir) / "csn_database" / "stats.json"
    if sf.exists():
        st = json.load(open(sf))
        md += ["## Pembersihan database retrieval (leakage)\n", "| Tahap | Jumlah |\n|---|---|"] + \
              [f"| {k} | {v} |" for k, v in st.items()] + [""]
    logs = plot_training(a.ckpt_dir, fdir)
    if logs:
        md += ["## Epoch terbaik per model (berdasarkan BLEU dev)\n", "| Model | Epoch terbaik | Epoch dijalankan | BLEU dev | F1 dev |\n|---|---|---|---|---|"]
        for n, L in logs.items():
            b = max(L, key=lambda r: r["dev_bleu"])
            md.append(f"| {n} | {b['epoch']} | {L[-1]['epoch']} | {b['dev_bleu']:.2f} | {b['dev_f1']:.2f} |")
        md.append("")
    plot_bars(res, fdir)
    has_ret = plot_retriever(a.ckpt_dir, fdir)
    md += ["## Grafik\n", "- `figures/training_curves.png` kurva loss dan BLEU dev per epoch",
           "- `figures/test_metrics.png` perbandingan metrik test" + ("\n- `figures/retriever.png` MRR dan Recall@K retriever" if has_ret else "")]
    (rdir / "results_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    q = qualitative(res, rdir, a.qual_exp, a.data_dir)
    if q:
        (rdir / "qualitative.md").write_text("\n".join(q[0]), encoding="utf-8")
    print(f"[OK] Ditulis ke {rdir}: results_tables.md, results_dev.csv, results_test.csv, figures/*.png"
          + (f", qualitative.md ({q[1]})" if q else ""))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results")
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--qual_exp", default=None, help="eksperimen untuk contoh kualitatif (default: F1 test terbaik)")
    main(p.parse_args())