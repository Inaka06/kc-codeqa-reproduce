# REPORT

## Perubahan kode
- preprocess.py: default FULL; tambah flag --smoke yang menulis ke processed_smoke/.
- smoke.py: baca processed_smoke/, simpan smoke.pt di processed_smoke/.
- data.py: tambah parameter root (default processed) agar smoke test tidak menimpa processed.
- processed/ kini berisi dataset penuh (Java 95778/12000/12000, Python 56085/7000/7000).

## Bukti verifikasi
- Java vocab_size=37670, max_src_len=417, max_tgt_len=23.
- Python vocab_size=50000, max_src_len=125, max_tgt_len=16.
- Smoke test: Java avg loss 5.3178, Python avg loss 5.3227 (500 sampel, 1 epoch), decode berjalan, checkpoint tersimpan di processed_smoke/.

## Asumsi
- Vocab Python dibatasi 50000 top-frekuensi.
- FFN dim 2048, grad clip 5, relative pos clip 32, dropout 0.2, lr 1e-4, patience 20, max 200 epoch, beam 4.
- Validasi menggunakan greedy (bukan beam 4) untuk kecepatan.
PY