import os
import io

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from model import CaptchaModel, load_model, DECODING_DICT, NORM_MEAN, NORM_STD

st.set_page_config(
    page_title="CAPTCHA OCR Predictor",
    page_icon="🔐",
    layout="centered",
)

MODEL_PATH = os.path.join("models", "captcha_model.ckpt")
MODEL_URL_ENV = "MODEL_URL"  # nama secret/env var berisi link download model


def ensure_model_downloaded():
    """
    Jika file model belum ada secara lokal (models/captcha_model.ckpt),
    coba download dari URL yang disimpan di Streamlit secrets (st.secrets["MODEL_URL"])
    atau environment variable MODEL_URL. Berguna karena file model biasanya
    terlalu besar untuk disimpan langsung di repo GitHub.
    """
    if os.path.exists(MODEL_PATH):
        return True

    model_url = None
    try:
        model_url = st.secrets.get(MODEL_URL_ENV)
    except Exception:
        pass
    if not model_url:
        model_url = os.environ.get(MODEL_URL_ENV)

    if not model_url:
        return False

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with st.spinner("Mengunduh model untuk pertama kali, mohon tunggu (bisa beberapa menit)..."):
        try:
            import gdown
            # gdown menangani file besar di Google Drive dengan benar
            # (melewati halaman peringatan "tidak bisa scan virus" untuk file >100MB)
            gdown.download(model_url, MODEL_PATH, quiet=False, fuzzy=True)

            if not os.path.exists(MODEL_PATH):
                raise RuntimeError("Download tidak menghasilkan file sama sekali.")

            size_bytes = os.path.getsize(MODEL_PATH)
            if size_bytes < 10_000_000:  # file model harusnya puluhan/ratusan MB
                # Kemungkinan besar yang terdownload adalah halaman HTML (error/login), bukan file asli
                with open(MODEL_PATH, "rb") as f:
                    head = f.read(200)
                os.remove(MODEL_PATH)
                raise RuntimeError(
                    f"File hasil download cuma {size_bytes} bytes (kemungkinan bukan file model asli, "
                    f"tapi halaman error dari Google Drive). Cuplikan awal file: {head[:100]!r}. "
                    "Pastikan sharing setting file di Drive sudah 'Anyone with the link'."
                )

            # Cek tanda file valid: torch.save (>=1.6) menyimpan sebagai file ZIP, harus diawali 'PK'
            with open(MODEL_PATH, "rb") as f:
                magic = f.read(4)
            if magic[:2] != b"PK":
                os.remove(MODEL_PATH)
                raise RuntimeError(
                    f"File yang terdownload ({size_bytes} bytes) bukan file checkpoint PyTorch yang valid "
                    f"(magic bytes: {magic!r}, seharusnya diawali 'PK'). File kemungkinan corrupt/tidak lengkap."
                )

            return True
        except Exception as e:
            st.error(f"Gagal mengunduh model: {e}")
            return False


@st.cache_resource(show_spinner="Memuat model ke memori...")
def get_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(MODEL_PATH, device=device)
    return model, device


def preprocess_image(image: Image.Image):
    transform = transforms.Compose(
        [
            transforms.Resize((50, 250)),
            transforms.CenterCrop((50, 250)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((NORM_MEAN,), (NORM_STD,)),
        ]
    )
    img = transform(image.convert("RGB"))
    if img.dim() == 3:
        img = img.unsqueeze(0)
    return img


def predict(model, device, image: Image.Image) -> str:
    img_tensor = preprocess_image(image).to(device)
    with torch.inference_mode():
        out = model(img_tensor)
    encoded_vector = out.reshape(21, 6).argmax(0)
    label = "".join(DECODING_DICT[int(idx)] for idx in encoded_vector.detach().cpu().numpy())
    return label


def main():
    st.title("🔐 CAPTCHA OCR Predictor")
    st.write(
        "Upload gambar CAPTCHA (6 karakter), dan model ResNet50 akan mencoba menebak teksnya."
    )

    if not ensure_model_downloaded():
        st.error(
            "File model tidak ditemukan di `models/captcha_model.ckpt` dan tidak ada "
            "`MODEL_URL` yang dikonfigurasi di Secrets. Silakan tambahkan file model "
            "atau atur secret `MODEL_URL` (lihat README)."
        )
        st.stop()

    model, device = get_model()

    uploaded_file = st.file_uploader(
        "Pilih gambar CAPTCHA", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(io.BytesIO(uploaded_file.read()))
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Gambar yang diupload", use_container_width=True)

        with st.spinner("Memprediksi..."):
            prediction = predict(model, device, image)

        with col2:
            st.metric(label="Prediksi CAPTCHA", value=prediction)

    st.divider()
    st.caption("Model: ResNet50 fine-tuned untuk klasifikasi 6-karakter CAPTCHA (21 kelas karakter).")


if __name__ == "__main__":
    main()
