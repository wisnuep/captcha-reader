# CAPTCHA OCR — Streamlit App

Aplikasi Streamlit untuk memprediksi teks CAPTCHA (6 karakter) menggunakan model ResNet50
yang sudah di-fine-tune (dari notebook `APLIKASI_CAPTCHA_UDAH_BENER_BLM_DEPLOY.ipynb`).

## Struktur folder

```
captcha-app/
├── app.py              # Aplikasi Streamlit utama
├── model.py             # Definisi arsitektur model + fungsi load
├── requirements.txt      # Dependencies
├── .gitignore
├── models/
│   └── captcha_model.ckpt   # <-- file model kamu taruh di sini (lihat langkah di bawah)
└── README.md
```

---

## LANGKAH 0 — Dapatkan file checkpoint model (WAJIB)

Aplikasi ini butuh file bobot model hasil training (`.ckpt` atau `.pth`). Notebook kamu
menyimpannya otomatis saat training di Colab, di path seperti:

```
/content/lightning_logs/version_0/checkpoints/epoch=34-step=2345.ckpt
```

Pilih salah satu skenario berikut:

### Skenario A — Kamu masih punya file ini (di Drive/laptop/Colab yang masih aktif)
1. Download file `.ckpt` tersebut.
2. Rename jadi `captcha_model.ckpt`.
3. Taruh di folder `models/` pada project ini.
4. Lanjut ke **LANGKAH 1**.

### Skenario B — File hilang, perlu training ulang di Colab
Tambahkan sel baru di paling akhir notebook kamu (setelah training selesai), lalu jalankan:

```python
import shutil
from google.colab import drive

drive.mount('/content/drive')

# Sesuaikan path best_model_path sesuai output cell training (checkpoint.best_model_path)
best_ckpt = checkpoint.best_model_path
shutil.copy(best_ckpt, '/content/drive/MyDrive/captcha_model.ckpt')
print("Model tersimpan di Google Drive:", '/content/drive/MyDrive/captcha_model.ckpt')
```

Setelah itu, download file dari Google Drive kamu, lalu ikuti Skenario A.

> Kenapa harus disalin ke Drive? Karena folder `/content/` di Colab bersifat sementara
> dan akan terhapus begitu runtime disconnect/reset.

---

## LANGKAH 1 — Ukuran file model & cara menyimpannya di GitHub

Model ResNet50 biasanya berukuran **~100-150 MB**, sementara GitHub membatasi file biasa
maksimal **100 MB**. Ada 2 opsi:

### Opsi 1 (paling gampang) — Git LFS
```bash
git lfs install
git lfs track "*.ckpt"
git add .gitattributes
```
Lalu commit file model seperti biasa (lihat Langkah 2). GitHub akan otomatis
menyimpannya lewat LFS.

### Opsi 2 — Download model saat aplikasi start (tidak simpan di repo)
1. Upload file `.ckpt` ke Google Drive, lalu klik kanan → **Get link** → set jadi "Anyone with the link".
2. Ambil FILE_ID dari link (bagian setelah `/d/` dan sebelum `/view`).
3. Buat direct-download link:
   ```
   https://drive.google.com/uc?export=download&id=FILE_ID
   ```
4. Saat deploy ke Streamlit Cloud, tambahkan secret `MODEL_URL` dengan value link di atas
   (lihat Langkah 4). App akan otomatis download saat pertama kali dijalankan.
5. Jangan lupa hapus komentar di `.gitignore` supaya folder `models/*.ckpt` tidak ikut ter-commit.

**Rekomendasi:** kalau file model < 100MB, pakai Opsi 1 (lebih simpel & stabil).
Kalau lebih besar / mau repo tetap ringan, pakai Opsi 2.

---

## LANGKAH 2 — Push ke GitHub

Dari folder project ini:

```bash
cd captcha-app
git init
git add .
git commit -m "Initial commit: CAPTCHA OCR Streamlit app"
git branch -M main
git remote add origin https://github.com/USERNAME/NAMA-REPO.git
git push -u origin main
```

Ganti `USERNAME/NAMA-REPO` dengan repo GitHub kamu (buat dulu repo kosong di github.com
kalau belum ada).

---

## LANGKAH 3 — Deploy ke Streamlit Community Cloud

1. Buka https://share.streamlit.io dan login pakai akun GitHub kamu.
2. Klik **"New app"**.
3. Pilih repo, branch (`main`), dan main file path: `app.py`.
4. Klik **Deploy**.

---

## LANGKAH 4 — (Kalau pakai Opsi 2) Set secret MODEL_URL

Di dashboard Streamlit Cloud:
1. Buka app kamu → klik **⋮ (titik tiga)** → **Settings** → **Secrets**.
2. Tambahkan:
   ```toml
   MODEL_URL = "https://drive.google.com/uc?export=download&id=FILE_ID"
   ```
3. Save. App akan otomatis restart dan mendownload model saat pertama kali dibuka.

---

## Testing lokal (opsional, sebelum deploy)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Buka `http://localhost:8501` di browser.

---

## Catatan penting

- Arsitektur model di `model.py` **harus identik** dengan yang di notebook training
  (ResNet50 dengan conv1 & fc yang dimodifikasi). Jangan diubah kecuali kamu juga
  mengubah & melatih ulang modelnya.
- `DECODING_DICT`, `NORM_MEAN`, `NORM_STD` di `model.py` juga harus sama persis dengan
  yang dipakai saat training (sudah disalin dari notebook kamu).
