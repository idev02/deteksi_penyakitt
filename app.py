import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import os
from PIL import Image

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Deteksi Penyakit Kuku & Tangan",
    layout="wide",
    page_icon="🩺"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ================== LOAD MODEL ==================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        os.path.join(BASE_DIR, "model", "hand_nail_disease_cnn.h5")
    )

model = load_model()

# ================== CLASS & INFO ==================
CLASS_NAMES = [
    "Acral_Lentiginous_Melanoma",
    "Healthy_Nail",
    "Onychogryphosis",
    "blue_finger",
    "clubbing",
    "pitting"
]

DISEASE_INFO = {
    "Acral_Lentiginous_Melanoma": "Kanker kulit langka pada kuku/telapak tangan dengan bercak gelap memanjang.",
    "Healthy_Nail": "Kondisi kuku sehat dan normal tanpa gangguan medis.",
    "Onychogryphosis": "Pertumbuhan kuku menebal dan melengkung menyerupai cakar.",
    "blue_finger": "Gangguan sirkulasi darah yang menyebabkan perubahan warna kebiruan pada jari.",
    "clubbing": "Pembesaran ujung jari akibat penyakit paru atau jantung kronis.",
    "pitting": "Cekungan kecil pada kuku yang sering berkaitan dengan psoriasis."
}

# ================== UI ==================
st.title("🩺 Sistem Deteksi Penyakit Kuku & Tangan")
st.markdown("Deteksi otomatis menggunakan **Convolutional Neural Network (CNN)**")

col1, col2 = st.columns(2)

with col1:
    uploaded = st.file_uploader("📷 Upload gambar kuku atau tangan", type=["jpg", "jpeg", "png"])

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Gambar yang diupload", width=350)

        img = np.array(image)
        img = cv2.resize(img, (224,224)) / 255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)[0]
        idx = np.argmax(pred)
        disease = CLASS_NAMES[idx]

        st.success(f"🧾 **Hasil Deteksi: {disease}**")
        st.write(f"Confidence: **{pred[idx]*100:.2f}%**")
        st.info(f"🩺 **Keterangan:** {DISEASE_INFO[disease]}")

with col2:
    st.subheader("📊 Confusion Matrix (Evaluasi CNN)")
    st.image("static/images/confusion_matrix.png", width=500)

st.markdown("---")

# ================== TABEL AKURASI ==================
st.subheader("📋 Tabel Evaluasi Akurasi CNN")

df = pd.read_csv("model/classification_report.csv")

best_class = df.loc[df["f1-score"].idxmax()].name
st.success(f"🏆 Penyakit dengan performa klasifikasi terbaik: **{best_class}**")

st.dataframe(df, use_container_width=True)

st.caption("Model dilatih dan dievaluasi menggunakan dataset citra penyakit kuku & tangan.")
