# realtime_app.py
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
from PIL import Image
import pathlib
import base64
from collections import OrderedDict 


# ==== Konfigurasi Halaman ====
st.set_page_config(page_title="Klasifikasi Jenis Beras", layout="wide")

# ==== Load CSS dari file ====
css_path = pathlib.Path("style.css")
if css_path.exists():
    with open(css_path, encoding='utf-8') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.error("File style.css tidak ditemukan.")


# ==== Navigation Bar ====
st.markdown("""
<div class="navbar">
    <a href="#beranda">Beranda</a>
    <a href="#prediksi-anchor">Klasifikasi</a>
</div>
""", unsafe_allow_html=True)

# ==== Hero Section ====
st.markdown("""
<div class="hero-section-wrapper">
    <div class="hero-section" id="beranda">
        <div class="hero-text">
            <h1>Klasifikasi Jenis Beras</h1>
            <p>Sistem klasifikasi ini mampu mengenali lima jenis beras: Pandan Wangi, Rojolele, IR64, Basmati, dan Ketan Putih. Model ini dibangun menggunakan algoritma Support Vector Machine (SVM) dengan pemrosesan fitur visual berupa warna dan bentuk untuk hasil klasifikasi yang cepat dan akurat.</p>
        </div>
        <div class="hero-image">
            <img src="https://github.com/baiq99/klasifikasi-beras/blob/main/img/icon-beras.png?raw=true" alt="Ilustrasi Beras">
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ==== Fitur Info Cards ====
st.markdown("""
<div class="info-heading">
    <h1>Jenis Beras Yang Digunakan Untuk Klasifikasi</h1>
</div>

<div class="info-section" id="fitur">
    <div class="info-card">
        <img src="https://github.com/baiq99/klasifikasi-beras/blob/main/img/pw.png?raw=true">
        <h4>Pandan Wangi</h4>
        <p>Beras lokal khas Cianjur yang berwarna putih dan beraroma pandan. Dikenal dengan rasa yang pulen dan banyak diminati karena kualitasnya yang unggul.</p>
    </div>
    <div class="info-card">
        <img src="https://github.com/baiq99/klasifikasi-beras/blob/main/img/r.png?raw=true">
        <h4>Rojolele</h4>
        <p>Beras unggulan asal Klaten dengan rasa nasi yang pulen dan enak. Memiliki bulir gemuk dan kaya nutrisi, banyak dikonsumsi oleh masyarakat Jawa Tengah.</p>
    </div>
    <div class="info-card">
        <img src="https://github.com/baiq99/klasifikasi-beras/blob/main/img/ir.png?raw=true">
        <h4>IR 64</h4>
        <p>Varietas beras produksi massal yang paling banyak ditanam di Indonesia. Bentuknya agak ramping dan rasanya pulen sedikit pera, cocok untuk berbagai masakan sehari-hari.</p>
    </div>
    <div class="info-card">
        <img src="https://github.com/baiq99/klasifikasi-beras/blob/main/img/b.png?raw=true">
        <h4>Basmati</h4>
        <p>Beras aromatik premium dengan bentuk ramping dan panjang. Setelah dimasak, teksturnya ringan dan memanjang hingga 2 kali lipat. Cocok untuk hidangan nasi khas India dan Timur Tengah.</p>
    </div>
    <div class="info-card">
        <img src="https://github.com/baiq99/klasifikasi-beras/blob/main/img/kp.png?raw=true">
        <h4>Ketan Putih</h4>
        <p>Beras berwarna putih dengan tekstur rapuh dan lengket. Mengandung pati tinggi dengan amilosa sangat rendah, cocok untuk kue tradisional dan makanan khas berbasis ketan.</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ==== Load Model ====
model = joblib.load("modelv4/svm_model_new.pkl")
scaler = joblib.load("modelv4/scaler_new.pkl")
class_labels = model.classes_.tolist()

# ==== Fungsi Preprocessing ====
def mask_largest_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return image
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    return cv2.bitwise_and(image, image, mask=mask)

def convertToHSV_withMask(image):
    masked = mask_largest_contour(image)
    return cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)

def extract_color_moments(hsv_masked):
    feats = []
    for ch in cv2.split(hsv_masked):
        c = ch.flatten().astype(np.float32)
        feats.extend([np.mean(c), np.std(c), np.mean((c - np.mean(c))**3) / (np.std(c)**3 + 1e-10)])
    return feats

def fullPreprocessingHu(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

def extract_hu_moments(image, min_area_threshold=50, top_n=3):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) >= min_area_threshold]
    if not valid: return [0.0] * 7
    mean_area = np.mean([cv2.contourArea(c) for c in valid])
    closest = sorted(valid, key=lambda c: abs(cv2.contourArea(c) - mean_area))[:top_n]
    hu_all = []
    for c in closest:
        m = cv2.moments(c)
        hu = cv2.HuMoments(m).flatten()
        hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
        hu_all.append(hu_log)
    return np.mean(hu_all, axis=0).tolist()

def predict_rice(image):
    resized = cv2.resize(image, (500, 500))
    hsv_masked = convertToHSV_withMask(resized)
    color_feats = extract_color_moments(hsv_masked)
    hu_input = fullPreprocessingHu(resized)
    hu_feats = extract_hu_moments(hu_input)
    combined = np.array(color_feats + hu_feats).reshape(1, -1)
    scaled = scaler.transform(combined)
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0]
    max_conf = np.max(prob) * 100
    prob_dict = {label: float(f"{p*100:.2f}") for label, p in zip(class_labels, prob)}
    return pred, prob_dict, max_conf

# ==== UI Prediksi ====
st.markdown('<div id="prediksi-anchor"></div>', unsafe_allow_html=True)
st.markdown("""
<div id="prediksi">
    <h1>Klasifikasi Jenis Beras</h1>
    <p>Unggah gambar beras Anda untuk mendapatkan hasil klasifikasi berdasarkan model SVM.</p>
</div>
""", unsafe_allow_html=True)

# Wrapper tengah
col_left, col_center, col_right = st.columns([1, 3, 1])
with col_center:
    uploaded = st.file_uploader("Pilih Gambar Beras", type=["jpg", "jpeg", "png"])

    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        # Gambar tampil di tengah dengan max-width 500px
        encoded = base64.b64encode(cv2.imencode('.jpg', image)[1]).decode()
        st.markdown(
            f"""
            <div class="image-preview">
                <img src="data:image/jpeg;base64,{encoded}" alt="Gambar yang Diunggah"/>
                <p class="caption">Gambar yang Diunggah</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Tombol di tengah
        st.markdown("<div class='btn-wrapper'>", unsafe_allow_html=True)
        if st.button("Klasifikasi Gambar"):
            pred, probs, conf = predict_rice(image)
            threshold = 44
            is_recognized = conf >= threshold
            result = pred if is_recognized else "Objek Tidak Dikenali"

            st.markdown(f"<div class='result-box'><strong>Jenis Beras:</strong> {result}</div>", unsafe_allow_html=True)

            if is_recognized:
                st.markdown(f"<div class='accuracy'><strong>Akurasi Prediksi:</strong> {conf:.2f}%</div>", unsafe_allow_html=True)
                st.markdown("<div class='centered-text'>Distribusi Probabilitas:</div>", unsafe_allow_html=True)

                st.markdown("<div class='table-wrapper'>", unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(probs.items(), columns=["Jenis Beras", "Probabilitas (%)"]), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)  # end table-wrapper

        st.markdown("</div>", unsafe_allow_html=True)  # end btn-wrapper

    st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)


