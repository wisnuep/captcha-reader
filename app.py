import os
import io

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from model import CaptchaModel, load_model, DECODING_DICT, NORM_MEAN, NORM_STD

st.set_page_config(
    page_title="Pembaca Kode Captcha",
    page_icon="🔐",
    layout="centered",
)

MODEL_PATH = os.path.join("models", "captcha_model.ckpt")
MODEL_URL_ENV = "MODEL_URL"  # nama secret/env var berisi link download model

CUSTOM_CSS = """
<style>
.main-header {
    text-align: center;
    padding: 1.2rem 0 0.4rem 0;
}
.main-header h1 {
    font-size: 2.2rem;
    margin-bottom: 0.2rem;
}
.main-header p {
    color: rgba(250, 250, 250, 0.65);
    font-size: 1rem;
}
.result-card {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.12), rgba(236, 72, 153, 0.10));
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 16px;
    padding: 1.6rem 1.6rem 1.2rem 1.6rem;
    margin-top: 0.6rem;
    margin-bottom: 1rem;
}
.pred-text {
    font-family: "Courier New", monospace;
    font-size: 2.8rem;
    font-weight: 700;
    letter-spacing: 0.35rem;
    text-align: center;
    margin: 0.2rem 0 0.8rem 0;
}
.overall-conf {
    text-align: center;
    font-size: 0.95rem;
    color: rgba(250, 250, 250, 0.7);
    margin-bottom: 1rem;
}
.char-row {
    display: flex;
    justify-content: center;
    gap: 0.5rem;
    flex-wrap: wrap;
}
.char-chip {
    display: flex;
    flex-direction: column;
    align-items: center;
    border-radius: 10px;
    padding: 0.5rem 0.7rem;
    min-width: 52px;
    color: white;
}
.char-chip .letter {
    font-family: "Courier New", monospace;
    font-size: 1.3rem;
    font-weight: 700;
}
.char-chip .pct {
    font-size: 0.72rem;
    opacity: 0.9;
    margin-top: 0.1rem;
}
.conf-high { background: rgba(34, 197, 94, 0.35); border: 1px solid rgba(34, 197, 94, 0.6); }
.conf-mid  { background: rgba(234, 179, 8, 0.30); border: 1px solid rgba(234, 179, 8, 0.55); }
.conf-low  { background: rgba(239, 68, 68, 0.30); border: 1px solid rgba(239, 68, 68, 0.55); }
</style>
"""


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
            try:
                gdown.download(model_url, MODEL_PATH, quiet=False, fuzzy=True)
            except TypeError:
                # Versi gdown yang lebih lama/berbeda tidak punya argumen 'fuzzy'
                gdown.download(model_url, MODEL_PATH, quiet=False)

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


def predict(model, device, image: Image.Image):
    """
    Mengembalikan (label, list_confidence_per_karakter).
    Confidence dihitung lewat softmax di antara 21 kemungkinan karakter,
    untuk masing-masing dari 6 posisi captcha.
    """
    img_tensor = preprocess_image(image).to(device)
    with torch.inference_mode():
        out = model(img_tensor)

    logits = out.reshape(21, 6)          # (kelas_karakter, posisi)
    probs = torch.softmax(logits, dim=0)  # softmax antar kelas karakter, per posisi
    conf_values, class_idx = probs.max(dim=0)

    label = "".join(DECODING_DICT[int(i)] for i in class_idx.detach().cpu().numpy())
    confidences = [float(c) for c in conf_values.detach().cpu().numpy()]
    return label, confidences


def confidence_class(pct: float) -> str:
    if pct >= 90:
        return "conf-high"
    if pct >= 70:
        return "conf-mid"
    return "conf-low"


def render_result(label: str, confidences: list):
    overall = sum(confidences) / len(confidences) * 100
    weakest = min(confidences) * 100

    chips_html = ""
    for ch, conf in zip(label, confidences):
        pct = conf * 100
        css_class = confidence_class(pct)
        chips_html += (
            f'<div class="char-chip {css_class}">'
            f'<span class="letter">{ch}</span>'
            f'<span class="pct">{pct:.0f}%</span>'
            f"</div>"
        )

    st.markdown(
        f"""
        <div class="result-card">
            <div class="pred-text">{label}</div>
            <div class="overall-conf">
                Confidence rata-rata: <b>{overall:.1f}%</b>
                &nbsp;•&nbsp; Karakter paling lemah: <b>{weakest:.1f}%</b>
            </div>
            <div class="char-row">{chips_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if weakest < 70:
        st.warning(
            "⚠️ Ada karakter dengan confidence rendah — ada kemungkinan salah tebak "
            "untuk karakter tersebut. Coba gunakan gambar yang lebih jelas kalau bisa."
        )


def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="main-header">
            <h1>🔐 Pembaca Kode Captcha</h1>
            <p>Upload gambar CAPTCHA 6 karakter — model ResNet50 akan menebak teksnya beserta tingkat keyakinannya.</p>
        </div>
        """,
        unsafe_allow_html=True,
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

        col1, col2 = st.columns([1, 1], gap="medium")
        with col1:
            st.image(image, caption="Gambar yang diupload", use_container_width=True)

        with st.spinner("Memprediksi..."):
            label, confidences = predict(model, device, image)

        with col2:
            st.markdown("**Hasil prediksi:**")
            render_result(label, confidences)
    else:
        st.info("👆 Upload gambar CAPTCHA (.jpg/.jpeg/.png) untuk mulai memprediksi.")

    st.divider()
    st.caption("Model: ResNet50 fine-tuned untuk klasifikasi 6-karakter CAPTCHA (21 kelas karakter).")


if __name__ == "__main__":
    main()
