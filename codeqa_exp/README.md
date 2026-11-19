# CodeQA Checkpoint 2: notebook eksperimen

Folder ini berisi semua eksperimen Checkpoint 2 dalam bentuk notebook, satu notebook per kelompok eksperimen.
Pertanyaan utamanya: **apakah model CodeQA benar-benar membaca kode, atau sebagian besar skornya datang dari pola pertanyaan dan jawaban?**
Metrik utama adalah **BERTScore F1**; BLEU, ROUGE-L, METEOR, EM, dan token F1 dilaporkan sebagai pendukung.

> Notebook di sini **belum dijalankan** (tidak ada output di dalamnya). Hasil run aslinya ada di server, di `codeqa_exp/runs/<tag>/results.json`.
> Rangkuman semua angka ada di spreadsheet "CodeQA Semua Metrik".

## Isi folder

```
codeqa_exp/
├── README.md                         dokumen ini
├── notebooks/                        11 notebook, urut sesuai alur eksperimen
└── code/
    ├── kc_common.py                  helper bersama: loader data + cek hash, template input, evaluasi, logging W&B
    ├── registry.py                   daftar folder run per model dan kondisi (dipakai notebook 11)
    └── nb_build.py                   generator: membuat ulang notebook dari skrip .py di server
codeqa_metrics.py, METRICS.md         (di root repo) evaluator resmi CodeQA yang diimpor kc_common.py
```

## Cara notebook dibuat

Setiap notebook dibuat otomatis oleh `code/nb_build.py` dari skrip Python yang benar-benar dipakai untuk run di server.
Kode skrip **disalin apa adanya** lalu dipecah per bagian (import, fungsi, blok utama). Satu-satunya perubahan adalah
`parser.parse_args()` menjadi `parser.parse_args(ARGS)`, supaya argumen bisa dipilih dari sel varian.
Generator memeriksa bahwa tidak ada baris yang hilang dan setiap sel bisa di-compile.

Skrip `.py` aslinya tidak ikut di repo ini (ada di server, `codeqa_exp/code/`). Logikanya lengkap di dalam notebook.
`nb_build.py` hanya perlu kalau ingin membuat ulang notebook setelah skripnya berubah.

## Struktur setiap notebook

1. **Judul dan penjelasan**: tujuan eksperimen, tabel varian, dan folder hasil.
2. **Setup**: menyamakan lingkungan dengan cara run aslinya (`bg.sh`): `cwd` di `codeqa_exp/code`, `pylib_metrics` di path,
   Java untuk METEOR, GPU 0.
3. **Sel varian** (`VARIANTS`, `RUN`, `ARGS`): daftar argumen persis seperti run yang sudah ada. Ganti `RUN`, restart kernel,
   lalu jalankan dari atas.
4. **Kode skrip**, satu bagian per sel, diawali docstring skripnya.
5. **Sel opsional `bg.sh`** (notebook training): menjalankan varian terpilih di tmux supaya training panjang tidak mati
   kalau browser atau kernel putus. Log masuk ke `logs/<RUN>.log`.
6. **Ringkasan hasil**: membaca `runs/<tag>/results.json` dan mencetak metrik test. Aman dijalankan kapan saja.

Notebook training memakai varian **`smoke`** sebagai default (beberapa langkah saja), supaya "Run All" tidak langsung
memulai training berjam-jam.

## Urutan dan ketergantungan

```
01 peta data ──────────────► 03 inferensi kode diacak/tetangga, 11 analisis
02 CodeT5 (checkpoint) ────► 03, 09 lintas domain, 10 audit split
09 bagian ext_convert ─────► 04 baseline (ClassEvalQA, CS1QA), 05 zero-shot
06, 07, 08 Qwen ───────────  berdiri sendiri
```

Checkpoint Seq2seq, CodeBERT-Transformer, dan CodeBERT-BART untuk notebook 03 berasal dari Checkpoint 1
(`prior_runs/kc-codeqa-reproduce/...`), bukan dilatih di sini.

## Penjelasan tiap notebook

### 01_data_maps: peta data bersama

Menyiapkan berkas peta yang dipakai notebook lain. Hanya CPU, tanpa argumen.

- **Kode diacak (E2)**: permutasi 7.000 contoh test dengan `random.Random(42)`, diulang sampai tidak ada contoh yang mendapat
  kodenya sendiri atau kode yang teksnya sama. Hasil: `maps/shuffled_mapping.json`.
- **Kode tetangga BM25 (E3)**: untuk setiap kode test, cari kode training paling mirip dengan BM25 (k1 = 1,5, b = 0,75, token
  huruf kecil per spasi), melewati kode yang identik atau Jaccard > 0,9. Hasil: `maps/neighbour_mapping.json`.
- **Leakage bins (E6)**: skor BM25 top-1 dinormalisasi lalu dibagi menjadi 5 kuintil (`maps/leakage_bins.json`).
- **ID studi metrik (E8)**: 300 contoh test terstratifikasi untuk studi metrik (`maps/metric_study_ids.json`).

### 02_codet5_protokol: CodeT5-small, protokol P

Satu skrip (`codet5_p.py`) untuk semua kondisi CodeT5-small. Protokol P: AdamW lr 5e-5, batch efektif 64, 20 epoch,
warmup 10%, clip 1,0, bf16, sumber 400 token, target 30 token, dev 7.000 beam 10, test beam 10.

| Varian | Kondisi | Seleksi checkpoint | Folder run |
|---|---|---|---|
| `full_v2`, `qonly_v1` | FULL / tanpa kode | dev BLEU | `runs/codet5_full_v2`, `runs/codet5_qonly_v1` |
| `full_bertsel`, `qonly_bertsel` | FULL / tanpa kode | dev BERTScore F1 | `runs/codet5_full_bertsel`, `runs/codet5_qonly_bertsel` |
| `shuffled_bertsel` | dilatih dengan kode diacak | dev BERTScore F1 | `runs/codet5_shuffled_bertsel` |
| `full_resplit_bertsel` | data digabung lalu dibagi ulang (seed 42) | dev BERTScore F1 | `runs/codet5_full_resplit_bertsel` |

Bagian kedua (`bertscore_codet5.py`) menghitung BERTScore untuk run seleksi BLEU.
Hasil acuan: FULL seleksi BERTScore memperoleh BERTScore F1 88,71 (rescaled 33,09); QONLY 85,57 (14,52).
Run resplit **tidak sebanding** dengan split resmi karena pasangan QA bocor antarsplit (lihat notebook 10).

### 03_inferensi_kode_diacak_tetangga: uji ketergantungan tanpa training ulang

Checkpoint FULL yang sudah ada di-decode ulang dengan kode test diganti:

- **shuffled**: kode milik contoh test lain (peta dari notebook 01);
- **neighbour**: kode training paling mirip menurut BM25;
- **full**: decode ulang tanpa perubahan, untuk memeriksa bahwa pipeline mereproduksi prediksi lama.

Model: Seq2seq (beam 4), CodeBERT-Transformer (beam 10), CodeBERT-BART (greedy dan beam 10), CodeT5-small (beam 10).
Template input, pemotongan, presisi, dan decoding tiap model tidak diubah; hanya string kodenya yang diganti.
16 varian; hasil di `runs/<model>__<kondisi>`.
Argumen Seq2seq dan CodeT5 sama persis dengan perintah aslinya. Argumen CodeBERT dan BART direkonstruksi dari
`results.json` run yang ada; `--group` dan `--protocol` hanya label W&B. Varian decode ulang FULL di sini tidak memakai
`--historical`, jadi pemeriksaan kecocokan dengan prediksi lama tidak ikut jalan.

### 04_baseline_tanpa_model: lantai skor

Dua baseline tanpa model untuk CodeQA Python test, Java test, ClassEvalQA, dan CS1QA:

- **Jawaban acak**: prediksi = jawaban referensi contoh lain di set yang sama, dengan teks berbeda (seed 42).
  Mengukur skor "jawaban bergaya CodeQA tapi untuk soal yang salah".
- **Echo pertanyaan**: prediksi = teks pertanyaannya sendiri.

Lalu keduanya dinilai dengan BERTScore (`runs/baselines/bertscore_baselines.json`).
Python test: jawaban acak 83,83 (rescaled 4,22), echo 82,91 (−1,27).

### 05_zero_shot: model tanpa fine-tuning

- **Qwen2.5-Coder-0.5B**: 4 set data × 2 mode (FULL, QONLY), prompt `question: …\ncode: …\nanswer:`, beam 10,
  maks 30 token baru, kode dipotong 350 token, baris pertama yang tidak kosong diambil sebagai jawaban.
- **CodeT5-small pretrained**: CodeQA Python test, FULL.
- **BERTScore** untuk prediksi zero-shot.

Python test: Qwen 82,23 (rescaled −5,27), CodeT5 71,44 (−69,21); keduanya di bawah baseline jawaban acak.
Hasil Qwen QONLY Python pernah terkena bug pemrosesan keluaran (`runs/zs_qwen_pythontest_qonly_old`) dan tidak dilaporkan.

### 06_qwen_full_finetune: Qwen2.5-Coder-0.5B, full fine-tuning

Prompt sama dengan zero-shot, loss hanya pada token jawaban (diakhiri EOS), lr 2e-5, batch efektif 64, maks 5 epoch,
checkpoint dipilih dengan dev BERTScore F1, test beam 10. Memakai `ft_qwen_proto.py`, yang identik dengan `ft_qwen.py`
kecuali dua argumen dev:

| Varian | Seleksi | Folder run |
|---|---|---|
| `full_dev1000_beam4` | 1.000 dev pertama, beam 4 (= run asli `ft_qwen.py`) | `runs/ft_qwen_python_full` |
| `full_dev7000_beam10` | seluruh 7.000 dev, beam 10 (protokol CodeT5) | `runs/ft_qwen_python_full_dev7000` |

Hasil `full_dev1000_beam4`: BERTScore F1 89,18 (rescaled 35,89).

### 07_qwen_lora: Qwen2.5-Coder-0.5B, LoRA (pengaturan awal)

LoRA r16, alpha 32, dropout 0,05 pada q, k, v, o, gate, up, down (8,8 juta parameter dilatih, 1,75%), lr 2e-4.
Input dan seleksi sama dengan notebook 06 varian dev 1.000 beam 4. Hasil: `runs/ft_qwen_python_lora`, BERTScore F1 89,18.

### 08_qwen_lora_protokol_codet5: LoRA dengan protokol CodeT5 (run acuan Qwen)

Konfigurasi LoRA sama dengan notebook 07, tetapi input dan seleksi disamakan dengan CodeT5-small: sumber 400 token
(`question + code`, lalu `\nanswer:`), target 30 token, seleksi dengan seluruh 7.000 dev beam 10, test beam 10.
Adapter tiap epoch disimpan (`epoch0` sampai `epoch4`).
Hasil: `runs/ft_qwen_python_lora_proto`, epoch terpilih 1, BERTScore F1 89,19 (rescaled 35,96), BLEU 36,22, EM 8,01.
Dev BERTScore memuncak di epoch 1 lalu turun walau training loss terus turun.

### 09_lintas_domain: CodeQA Java, ClassEvalQA, CS1QA

1. **`ext_convert.py`**: mengubah ClassEvalQA dan CS1QA (dataset `datapaf` di Hugging Face) ke format CodeQA
   (`test.code`, `test.question`, `test.answer` di `ext_data/`). Normalisasi kode didekati dengan regex: komentar dan
   docstring dihapus, identifier dipisah, string literal dipertahankan.
2. **`xdomain_codet5.py`**: model CodeT5-small Python (seleksi BERTScore) diterapkan tanpa training ulang, kondisi FULL,
   QONLY, dan kode diacak (9 varian).
3. **`bertscore_xdomain.py`**: BERTScore semua prediksi. Varian `java` menulis ke berkas baru supaya hasil lama tidak tertimpa.
4. **`xl_overlap.py`**: tumpang tindih string antara data Java dan Python CodeQA.

Pada Java, model masih di atas baseline dan kode menyumbang sekitar 55% skor di atas jawaban acak. Pada ClassEvalQA dan
CS1QA, model di bawah kedua baseline karena referensinya kalimat panjang sementara model menjawab frasa pendek.

### 10_audit_split: audit pembagian data

`split_audit.py` menghitung pengulangan pasangan QA dan jawaban di dalam setiap split resmi (Python dan Java),
tumpang tindih antarsplit, lalu memecah BERTScore run resplit (notebook 02) menjadi contoh yang bocor dan yang bersih.
Temuan: train resmi penuh pengulangan (Python 29% pasangan QA berulang) sedangkan dev/test hampir tidak; pada resplit,
21% pasangan QA test muncul di train. Varian `tanpa_bertscore` melewati perhitungan BERTScore.
Hasil: `runs/split_audit.json`.

### 11_analisis: analisis gabungan

- **`analysis.py`**: membaca `results.json` dan prediksi semua run (lewat `registry.py`): sensitivitas metrik per model dan
  kondisi, paired bootstrap (1.000 resample, seed 42), breakdown yes/no, dev BLEU vs dev F1, dan kuintil kebocoran.
- **`length_audit.py`**: audit panjang input test per keluarga tokenizer (E7).
- **`metric_study.py`**: persiapan studi metrik pada 300 contoh terstratifikasi (E8); tidak ada pelabelan.

Ketiga skrip tidak punya argumen dan langsung menulis berkas saat dijalankan.

## Menjalankan

Notebook ditulis untuk server UB AI Hub (`node680`), dengan kernel Python dari `~/venvs/dq`. Path di dalamnya mengarah ke
`~/inahuy/kacuy/...`: data CodeQA di `kc-codeqa-reproduce/data`, data eksternal di `ext_data`, dan hasil di `codeqa_exp/runs`.
Di mesin lain, sesuaikan dulu:

- path `EXP`/`REPO` (bisa lewat environment `KC_EXP` dan `KC_REPO` untuk `kc_common.py`) dan path yang ditulis langsung di
  beberapa skrip (`/home/nafisnaufal1426/...`);
- data CodeQA, `ext_data`, dan checkpoint Checkpoint 1 di `prior_runs/`;
- paket `torch`, `transformers`, `peft`, `bert_score`, `wandb`, `pandas`, `scipy`, serta Java untuk METEOR.

Semua run mencatat ke Weights & Biases (project `codeqa-kc`). API key tidak ditulis di kode; login lewat `wandb login`
atau environment.

## Catatan

- Jangan jalankan varian `full_dev7000_beam10` di notebook 06 selama run tmux `ft_proto` di server masih berjalan:
  keduanya menulis ke folder yang sama.
- Notebook 01, 04, 09 (`ext_convert`), 10, dan 11 menulis ulang berkas hasil begitu dijalankan. Skripnya deterministik
  (seed 42), jadi isinya sama selama data masukannya sama.
- BERTScore: `roberta-large` layer 17, bahasa Inggris, tanpa rescaling. Nilai rescaled =
  (F − b) / (1 − b), dengan b = 0,83122575 untuk F1
  ([sumber](https://github.com/Tiiiger/bert_score/blob/master/bert_score/rescale_baseline/en/roberta-large.tsv)).
- Token F1: huruf kecil, dipecah per spasi, token berulang tetap dihitung, dirata-rata per contoh (`codeqa_metrics.py`).
