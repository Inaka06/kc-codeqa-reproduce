CodeQA untuk VPS

Direktori kerja: ~/k15/seq2seq/kc-codeqa-reproduce
Environment: source ~/k15/.venv/bin/activate

Notebook dan runner

- CodeQA_Seq2Seq_Paper.ipynb: reproduksi Seq2seq.
- CodeQA_CodeBERT_Paper.ipynb dan codebert_training.py: reproduksi CodeBERT–Transformer.
- CodeQA_CodeBERT_BART_VPS.ipynb dan codebert_bart_training.py: eksperimen frozen CodeBERT dengan decoder BART pretrained.
- make_report_figures.py: ekspor grafik epoch dan tabel perbandingan untuk laporan; membaca JSON tanpa memuat model.
- CodeQA_CodeBERT_BART_Finished.ipynb: akan dibuat oleh nbconvert setelah eksekusi aktif selesai normal.

Data dan source rujukan

- data/python/: dataset Python resmi, pembagian train/dev/test.
- references/: source model dan provenance dari CodeQA serta NeuralCodeSum.
- source/: evaluator CodeQA.
- requirements-codebert.txt: dependency untuk environment CodeBERT/BART. Matplotlib tersedia pada Python sistem untuk script grafik.

Hasil eksperimen

- seq2seq_paper_results/: checkpoint, riwayat, hasil dan notebook report Seq2seq.
- codebert_paper_results/: checkpoint, riwayat, hasil dan notebook report CodeBERT–Transformer.
- codebert_bart_results/: checkpoint, riwayat dan hasil eksperimen CodeBERT–BART.
- report_figures/: PNG/PDF grafik, CSV tabel, report.md dan manifest snapshot; bisa dibuat ulang.

Direktori hasil juga memuat smoke test. Untuk hasil utama, pilih paper_v1_* pada Seq2seq/CodeBERT dan vps_v1_* pada hybrid. Jangan memindahkan atau menghapus direktori run yang sedang digunakan.

Perintah sehari-hari (dijalankan dari root proyek)

Perbarui grafik: /usr/bin/python3 make_report_figures.py
Pantau hybrid: tail -F codebert_bart_results/vps_v1_*/training.log
Baca hasil CodeBERT: cat codebert_paper_results/paper_v1_*/results.json
Panduan training dan resume hybrid: docs/CodeBERT_BART_README.md

Pengujian CPU (tanpa memakai GPU sharing)

CUDA_VISIBLE_DEVICES="" python tests/test_codebert_training.py
CUDA_VISIBLE_DEVICES="" python tests/test_codebert_bart_training.py
/usr/bin/python3 tests/test_make_report_figures.py

Dokumentasi

- docs/CodeBERT_BART_README.md: panduan eksperimen hybrid. Perintah test di panduan memakai tests/test_codebert_bart_training.py.
- docs/README_before_cleanup.md: README lama, dipertahankan sebagai arsip. Perintah test lama sekarang berada di tests/.
- docs/cleanup_plan.txt: cakupan cleanup dan batasannya.

Catatan struktur

Notebook dan runner tetap di root: source hash menjadi bagian identitas run dan nbconvert hybrid masih aktif. Perintah training sebelumnya tetap berlaku. Dataset dan hasil tidak dipindah atau dihapus. Cleanup ini hanya dilakukan pada VPS; belum disinkronkan ke laptop atau GitHub.
