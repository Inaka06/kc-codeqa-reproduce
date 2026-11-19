import os
import json
from pathlib import Path
import pandas as pd

def generate_report():
    metrics_dir = Path("runs/metrics")

    # Load dataset stats
    stats_path = metrics_dir / "dataset_stats.json"
    dataset_stats = {}
    if stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as f:
            dataset_stats = json.load(f)

    # Load test greedy metrics if available
    test_greedy_path = metrics_dir / "test_greedy_metrics.json"
    test_greedy = {}
    if test_greedy_path.exists():
        with open(test_greedy_path, "r", encoding="utf-8") as f:
            test_greedy = json.load(f)

    os.makedirs("runs/tables", exist_ok=True)
    os.makedirs("runs/figures", exist_ok=True)

    # Create tables
    # Table 1: Dataset statistics vs Paper
    stats_data = dataset_stats["stats"]  # error jelas kalau prepare_data.py belum dijalankan

    df_stats = pd.DataFrame(stats_data).T
    df_stats.to_csv("runs/tables/dataset_statistics.csv")
    df_stats.to_markdown("runs/tables/dataset_statistics.md")
    with open("runs/tables/dataset_statistics.tex", "w", encoding="utf-8") as f:
        f.write(df_stats.to_latex())

    # Table 2: Main evaluation results vs Paper baselines
    paper_baselines = [
        {"Model": "Seq2seq", "BLEU": 30.09, "ROUGE-L": 21.69, "METEOR": 6.38, "EM": 2.84, "F1": 23.27},
        {"Model": "Dual Encoder", "BLEU": 28.90, "ROUGE-L": 17.79, "METEOR": 4.53, "EM": 2.22, "F1": 18.97},
        {"Model": "Transformer", "BLEU": 30.69, "ROUGE-L": 23.26, "METEOR": 7.80, "EM": 3.35, "F1": 24.59},
        {"Model": "CodeBERT", "BLEU": 34.86, "ROUGE-L": 30.28, "METEOR": 12.51, "EM": 4.93, "F1": 31.56},
        {"Model": "Human (100 samples)", "BLEU": 55.96, "ROUGE-L": 59.55, "METEOR": 25.49, "EM": 32.00, "F1": 61.59}
    ]

    qwen_metrics = test_greedy["metrics"]  # error jelas kalau evaluate.py belum dijalankan
    qwen_row = {"Model": "Qwen2.5-Coder-0.5B + LoRA (Greedy)", **qwen_metrics}

    all_results = paper_baselines + [qwen_row]
    df_results = pd.DataFrame(all_results)
    df_results.to_csv("runs/tables/test_results.csv", index=False)
    df_results.to_markdown("runs/tables/test_results.md", index=False)
    with open("runs/tables/test_results.tex", "w", encoding="utf-8") as f:
        f.write(df_results.to_latex(index=False))

    # Generate REPORT.md
    report_content = f"""# Laporan Reproduksi CodeQA (Python) — Qwen2.5-Coder-0.5B + LoRA

## 1. Tujuan Eksperimen
Mereproduksi pipeline Question Answering untuk pemahaman kode sumber menggunakan dataset CodeQA Python dan base model Qwen2.5-Coder-0.5B dengan fine-tuning parameter-efficient LoRA.

## 2. Setup Lingkungan & Konfigurasi
- **Base Model**: `Qwen/Qwen2.5-Coder-0.5B` (Causal LM, bf16, sdpa)
- **Metode LoRA**: r=16, alpha=32, dropout=0.05, target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`.
- **Hiperparameter Training**: AdamW, lr 2e-4, weight decay 0.01 (LoRA), batch efektif 64 (micro-batch + gradient accumulation), 10 epoch, early stopping patience 3 epoch berdasarkan BLEU dev.
- **Token Length max_length**: P99 train set dibulatkan ke kelipatan 64 (`{dataset_stats.get('max_length', 512)}`).

## 3. Statistik Dataset
{df_stats.to_markdown()}

## 4. Hasil Evaluasi Akhir (Test Set)
{df_results.to_markdown(index=False)}

## 5. Breakdown Per Tipe Pertanyaan
"""

    breakdown = test_greedy.get("breakdown", {})
    if breakdown:
        # Separate question types vs length bins
        q_types = ["What", "How", "Where", "When", "Why", "For what purpose", "Yes/No", "Other"]
        qb = {k: v for k, v in breakdown.items() if k in q_types}
        lb = {k: v for k, v in breakdown.items() if k not in q_types}

        if qb:
            report_content += "### Breakdown Berdasarkan Tipe Pertanyaan\n"
            report_content += pd.DataFrame(qb).T.to_markdown() + "\n\n"
        if lb:
            report_content += "### Breakdown Berdasarkan Panjang Jawaban & Kode\n"
            report_content += pd.DataFrame(lb).T.to_markdown() + "\n\n"
    else:
        report_content += "Belum menggenerate data breakdown (jalankan evaluasi penuh).\n\n"

    report_content += """## 6. Analisis & Kesimpulan
- Perbandingan angka dengan paper menunjukkan performa Qwen2.5-Coder-0.5B + LoRA pada dataset Python CodeQA.
- Reproduksibilitas dijamin dengan konfigurasi seed tetap, snapshot konfigurasi, dan skrip otomatis.
"""

    report_path = "runs/REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"REPORT.md generated successfully at {report_path}")

if __name__ == "__main__":
    generate_report()