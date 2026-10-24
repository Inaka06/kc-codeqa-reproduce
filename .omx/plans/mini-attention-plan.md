# Rencana CodeQA Mini yang menghasilkan jawaban berguna

Status: rencana saja. Tidak ada implementasi atau training baru yang diizinkan oleh tahap perencanaan ini.

## Tujuan dan batas

- Model dapat dilatih dan dipakai pada laptop RTX 3050 dengan VRAM 4 GB.
- Anggaran seluruh percobaan training maksimal dua jam; jangan memakai seluruh anggaran jika target sudah tercapai.
- Menghasilkan beberapa jawaban test yang benar, bukan hanya membuktikan loss turun.
- Fokus pada Python dan dataset resmi yang sudah tersedia.
- Pertahankan `CodeQA_Mini.ipynb` dan hasil lamanya. Buat notebook baru hanya pada tahap implementasi nanti.
- Eksperimen ini merupakan adaptasi model kecil, bukan reproduksi arsitektur atau skor CodeBERT paper.

## Bukti kondisi awal

- `CodeQA_Mini.ipynb`, bagian konfigurasi/model/training: 256 train, 32 dev, 32 test; GRU encoder-decoder tanpa attention; 415.312 parameter; CPU; delapan epoch.
- `mini_codeqa_results/results.json`: EM 0 dan token F1 sekitar 10,11 pada 32 test. Training loss turun tetapi validation loss terbaik muncul pada epoch awal.
- Screenshot pengguna menunjukkan lima prediksi pertama semuanya `a`.
- Sebab kegagalan belum diisolasi. Data sedikit, bottleneck representasi encoder, serta pemilihan checkpoint merupakan hipotesis, bukan kesimpulan kausal.

## Rancangan eksperimen

### 1. Tetapkan data dan penilaian sebelum training

- Kandidat awal: 10.000 train, 500 dev, dan 500 test, masing-masing dari split resmi Python yang sesuai.
- Gunakan seed tetap dan simpan ID subset. Test tidak dipakai memilih model atau parameter.
- Periksa overlap string kode antar-subset. Jangan memindahkan contoh test ke train; laporkan overlap yang ditemukan.
- Vocabulary hanya dibuat dari train, kandidat batas 8.000 kata. Laporkan persentase token jawaban di luar vocabulary.
- Kandidat batas input 160 token dan target 24 token. Laporkan tingkat truncation, khususnya jawaban.
- Semua angka di atas adalah konfigurasi awal yang boleh diperkecil berdasarkan benchmark memori/waktu sebelum melihat test.

### 2. Bandingkan dua varian yang sebanding

- A: GRU tanpa attention, dilatih dengan data lebih banyak.
- B: GRU dengan attention ke representasi token input, sehingga decoder dapat mengakses bagian input saat membentuk jawaban.
- Gunakan tokenizer, vocabulary, subset, ukuran hidden/embedding, batas panjang, serta batas langkah training yang sama sejauh perbedaan arsitektur memungkinkan.
- Target ukuran di bawah lima juta parameter; jumlah sebenarnya harus dihitung sebelum training.
- Tanpa CodeBERT, distilasi, retrieval jawaban, atau jawaban hardcoded pada percobaan ini.
- Attention adalah perubahan metode; peningkatan kualitas belum dijamin. Referensi implementasi konsep: https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

### 3. Benchmark dan training berbatas waktu

- Coba satu forward/backward dan generation pada CPU serta GPU. Gunakan GPU jika tersedia dan konfigurasi muat.
- Ukur sekitar 100 minibatch setelah warmup, termasuk sinkronisasi CUDA saat mengukur waktunya.
- Mulai dari batch 32; perkecil jika memori tidak cukup. Ukur peak VRAM dan throughput.
- Proyeksikan biaya dua varian beserta validasi. Jika melebihi dua jam, kurangi jumlah langkah/epoch sebelum menjalankan percobaan penuh.
- Kandidat maksimum 20 epoch per varian, early stopping berdasarkan dev; catat jumlah langkah aktual jika berhenti lebih awal.
- Pilih checkpoint dari F1 jawaban hasil generation di dev, dengan loss dev sebagai pemecah seri. Jangan memakai test untuk memilih checkpoint.
- Simpan checkpoint, konfigurasi, kurva train/dev, durasi, dan seed. Tinjau distribusi prediksi dev untuk mendeteksi pengulangan jawaban.

### 4. Evaluasi sekali pada test setelah konfigurasi dipilih

- Jalankan kedua checkpoint terpilih pada 500 test yang sama.
- Laporkan EM, token F1, jumlah jawaban unik, jawaban kosong, dan porsi prediksi paling sering.
- Tampilkan beberapa contoh pertama tanpa seleksi, contoh benar, serta contoh gagal. Simpan seluruh prediksi agar contoh sukses tidak menjadi satu-satunya bukti.
- Pisahkan exact match dari penilaian manual atas parafrasa yang benar.
- Hitung keberhasilan non-Yes/No agar jawaban benar tidak hanya berasal dari tebakan pertanyaan biner.
- Jika anggaran habis atau target tidak tercapai, laporkan hasil apa adanya; jangan mengganti test atau menambal jawaban tertentu.

## Target keberhasilan

- Notebook baru dapat dijalankan dari awal sampai akhir tanpa OOM pada konfigurasi laptop yang diuji.
- Total training kedua varian tidak melebihi dua jam.
- Target praktis: setidaknya lima jawaban test exact match, termasuk jawaban non-Yes/No. Ini sasaran eksperimen, bukan jaminan sebelum training.
- Prediksi tidak seluruhnya menjadi satu jawaban; laporkan konsentrasi prediksi secara numerik.
- Kualitas dinilai pada test terpisah dan dibandingkan dengan varian A pada subset yang sama.
- Checkpoint yang dimuat ulang menghasilkan prediksi yang sama dengan sebelum disimpan.

## Verifikasi implementasi nanti

- Uji alignment tiga berkas dataset, determinisme subset, token khusus, dan penanganan kata baru.
- Uji attention tidak memberi bobot pada padding dan menghasilkan bobot yang jumlahnya satu pada input valid.
- Uji loss tidak menghitung padding, gradient finite, serta generation berhenti pada EOS atau batas panjang.
- Overfit beberapa contoh train sebagai diagnosis implementasi; keberhasilan pada contoh tersebut tidak dihitung sebagai keberhasilan test.
- Uji simpan/muat checkpoint dan jalankan semua sel notebook pada konfigurasi terpilih.

## Risiko dan mitigasi

- Attention saja mungkin tidak cukup: bandingkan A/B dan laporkan kegagalan; jangan mengklaim sebab tunggal dari screenshot.
- Data lebih banyak dapat menaikkan waktu: ukur lebih dahulu dan batasi total langkah.
- Jawaban umum dapat memberi F1 meskipun tidak informatif: laporkan EM, konsentrasi jawaban, serta contoh non-Yes/No.
- Bottleneck vocabulary dapat menghalangi jawaban benar: ukur OOV dan truncation sebelum memilih konfigurasi.
- Hasil ini tidak menggantikan reproduksi evaluasi CodeBERT resmi dalam tujuan intern reproduce–rethink–reinvent.

## Berkas yang akan disentuh pada tahap implementasi

- Baru: `CodeQA_Mini_Attention.ipynb`.
- Baru: direktori hasil khusus eksperimen attention dan baseline pembanding.
- Tetap sebagai referensi: `CodeQA_Mini.ipynb`, `mini_codeqa_results/`, dan `data/python/`.
