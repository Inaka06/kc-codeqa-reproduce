# CodeQA Seq2seq untuk VPS

Paket ini berisi notebook, dataset Python, source NeuralCodeSum, dan evaluator CodeQA. Jalankan notebook dari direktori ini agar dataset dan source ditemukan. Notebook merekonstruksi baseline Seq2seq berdasarkan paper dan source terkait.

Gunakan Python 3.10 atau 3.11 dengan Jupyter, PyTorch yang mendukung GPU/CUDA VPS, numpy, nltk, tqdm, prettytable, dan nbformat. Evaluasi METEOR membutuhkan Java (`java` di PATH). Pilih kernel environment tersebut dan pastikan GPU tersedia sebelum training.

`PROFILE = "paper"` menggunakan seluruh dataset Python, maksimal 200 epoch, patience 20. Batas satu sesi 2 jam (`MAX_SESSION_SECONDS`); untuk melanjutkan, jalankan ulang notebook dengan konfigurasi dan `RUN_TAG` sama serta `RESUME = True`. Hasil dan checkpoint tersimpan di `seq2seq_paper_results/`, dan tidak masuk Git. Evaluasi test penuh hanya berjalan setelah training selesai. Gunakan `PROFILE = "smoke"` untuk uji awal.

Jika server dipakai bersama, periksa GPU yang tersedia dan pilih satu GPU melalui `CUDA_VISIBLE_DEVICES` sebelum memulai kernel. Profil paper belum diuji pada VPS ini.
