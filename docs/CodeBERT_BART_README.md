# Eksperimen CodeBERT–BART di VPS

CodeBERT frozen → LayerNorm + Linear → pretrained BART-base decoder. Hanya bridge dan decoder dilatih; ini pengembangan eksperimental, bukan reproduksi baseline paper. Library, dataset Python, dan evaluator menggunakan paket CodeBERT sebelumnya (`requirements-codebert.txt`); METEOR membutuhkan Java.

Preset `vps`: 56.085 train, 1.000 dev tetap (seed 43), 7.000 test setelah selesai, maksimum 20 epoch, patience 5, source 400, target 32, effective batch 64, microbatch 2, LR 5e-5 dan warmup 10%. Dev/test greedy; baseline sebelumnya memakai beam 10. `full_paper_training_completed` selalu false karena arsitektur ini bukan baseline paper. Tidak ada batas sesi secara default. Dua GPU tidak digabung; pilih satu GPU yang tersedia.

```bash
cd ~/k15/seq2seq/kc-codeqa-reproduce
source ~/k15/.venv/bin/activate
nvidia-smi
tmux new -s codebert-bart
# Contoh GPU 1: jalankan setelah memori GPU tersebut cukup tersedia.
CUDA_VISIBLE_DEVICES=1 python -u codebert_bart_training.py --profile smoke --run-tag check --micro-batch 2
CUDA_VISIBLE_DEVICES=1 python -u codebert_bart_training.py --profile vps --run-tag v1 --micro-batch 2
```

Detach: Ctrl+B lalu D. Kembali: `tmux attach -t codebert-bart`. Resume setelah server mati/interupsi: perintah training yang sama. Jangan mengubah microbatch untuk tag yang sudah berjalan; konfigurasi berubah memerlukan tag baru. Jangan memulai penulis kedua untuk tag sama.

Log, benchmark, checkpoint, history dan hasil berada di `codebert_bart_results/vps_v1_<hash>/`. Benchmark setelah 10 update memberi estimasi per epoch, tidak termasuk dev/IO. Lama total belum diukur untuk preset ini. Checkpoint setiap sekitar lima menit dan akhir epoch; interupsi mengulang bagian setelah checkpoint terakhir. BART diunduh saat penggunaan pertama; cache CodeBERT sebelumnya dipakai lagi.

Notebook `CodeQA_CodeBERT_BART_VPS.ipynb` menggunakan runner sama. Melihat hasil tidak perlu menjalankan training lagi:

```bash
cat codebert_bart_results/vps_v1_*/status.json
cat codebert_bart_results/vps_v1_*/results.json
```

Validasi ringan CPU: `python -u tests/test_codebert_bart_training.py`. Dataset dan checkpoint reproduksi sebelumnya tidak diubah. Eksperimen laptop tetap terpisah; checkpoint laptop tidak kompatibel dengan preset VPS ini.
