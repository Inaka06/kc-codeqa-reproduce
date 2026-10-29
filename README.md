# CodeQA Seq2seq untuk VPS

Paket ini berisi notebook, dataset Python, source NeuralCodeSum, dan evaluator CodeQA. Jalankan notebook dari direktori ini agar dataset dan source ditemukan. Notebook merekonstruksi baseline Seq2seq berdasarkan paper dan source terkait.

Gunakan Python 3.10 atau 3.11 dengan Jupyter, PyTorch yang mendukung GPU/CUDA VPS, numpy, nltk, tqdm, prettytable, dan nbformat. Evaluasi METEOR membutuhkan Java (`java` di PATH). Pilih kernel environment tersebut dan pastikan GPU tersedia sebelum training.

`PROFILE = "paper"` menggunakan seluruh dataset Python, maksimal 200 epoch, patience 20. Batas satu sesi 2 jam (`MAX_SESSION_SECONDS`); untuk melanjutkan, jalankan ulang notebook dengan konfigurasi dan `RUN_TAG` sama serta `RESUME = True`. Hasil dan checkpoint tersimpan di `seq2seq_paper_results/`, dan tidak masuk Git. Evaluasi test penuh hanya berjalan setelah training selesai. Gunakan `PROFILE = "smoke"` untuk uji awal.

Jika server dipakai bersama, periksa GPU yang tersedia dan pilih satu GPU melalui `CUDA_VISIBLE_DEVICES` sebelum memulai kernel. Profil paper belum diuji pada VPS ini.

## CodeBERT + decoder Transformer

Notebook baru: `CodeQA_CodeBERT_Paper.ipynb`. Runner yang dipakai notebook dan terminal: `codebert_training.py`. Install `requirements-codebert.txt` ke environment PyTorch CUDA yang dipilih. Default satu GPU, 20 epoch, source 400, target 30, beam 10, effective batch 64, microbatch 4, FP16 dan encoder gradient checkpointing. Ini adaptasi memori dari baseline resmi; perubahan numerik dicatat di notebook/config.

Uji pertama: `python -u codebert_training.py --profile smoke --run-tag check --micro-batch 1`. Training penuh: `python -u codebert_training.py --profile paper --run-tag v1 --micro-batch 4`. Jalankan di tmux agar tidak bergantung pada laptop. Tidak ada batas dua jam secara default. Resume memakai perintah dan konfigurasi sama; log menampilkan `RESUME confirmed`. Model, optimizer, scheduler, scaler, dan RNG tersimpan. Profil/tag yang sedang berjalan dikunci untuk mencegah dua penulis checkpoint.

Checkpoint, log dan hasil berada di `codebert_paper_results/`, terpisah dari Seq2seq dan diabaikan Git. Source model + preprocessing resmi disertakan di `references/CodeQA_CodeBERT/` dengan commit, hash dan compatibility patch. Dataset dan evaluator memakai berkas yang sudah ada di paket ini. Training penuh pada GPU VPS belum diverifikasi.
