# app.py
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tempfile, os
from ecg_utils import bandpass_filter, zscore, compute_hr_hrv, estimate_intervals
import cv2
from ecg_digitizer_advanced import extract_leadII_signal_with_overlay
from pdf2image import convert_from_path

def pdf_to_image(pdf_path, dpi=300):
    pages = convert_from_path(pdf_path, dpi=dpi, poppler_path=r"C:\poppler\bin")
    return pages[0]

st.set_page_config(page_title="ECG Health Predictor", page_icon="❤️", layout="wide")

# ---- Constants ----
FS = 360
WINDOW = 1800  # 5s

@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("models/ecg_model.h5")
    except Exception as e:
        st.error("Model not found. Please run train_mitbih.py to create models/ecg_model.h5")
        st.stop()

model = load_model()

st.title("❤️ ECG Health Predictor")
st.markdown("Upload a CSV with an `ecg` column (single-lead waveform). Sampling rate assumed **360 Hz**.")

uploaded = st.file_uploader("Upload ECG (CSV,Image, or PDF)", type=["csv", "png", "jpg", "jpeg", "pdf"])

with st.expander("Format help", expanded=False):
    st.code("ecg\n0.014\n0.022\n...")

def predict_window(sig):
    sig = bandpass_filter(sig, fs=FS, low=0.5, high=40)
    sig = zscore(sig)
    x = sig[:WINDOW].reshape(1, WINDOW, 1)
    p = float(model.predict(x, verbose=0)[0][0])
    label = "⚠️ Abnormal (possible arrhythmia)" if p > 0.5 else "✅ Normal"
    return label, p, sig

if uploaded:
    ext = os.path.splitext(uploaded.name)[-1].lower()

    # CSV
    if ext == ".csv":
        df = pd.read_csv(uploaded)
        if 'ecg' not in df.columns:
            st.error("CSV must contain 'ecg' column")
            st.stop()
        raw = df['ecg'].astype(float).values

    # PDF -> image -> signal
    elif ext == ".pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.read())
            tmp.flush()
            img = pdf_to_image(tmp.name)
            img_path = tmp.name.replace(".pdf", ".png")
            img.save(img_path)
            img_bgr = cv2.imread(img_path)
            raw, overlay, dbg = extract_leadII_signal_with_overlay(img_bgr, fs=360, target_len=WINDOW)

    elif ext in [".png", ".jpg", ".jpeg"]:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(uploaded.read())
            tmp.flush()
            img_bgr = cv2.imread(tmp.name)
            raw, overlay, dbg = extract_leadII_signal_with_overlay(img_bgr, fs=360, target_len=WINDOW)
            
    else:
        st.error("Unsupported file type")
        st.stop()

    if raw is None or len(raw) < WINDOW:
        st.error("Failed to extract ECG signal. Try a cleaner scan.")
        st.stop()
    if overlay is not None:
        st.subheader("Digitizer Debug View")
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),caption=f"Lead II panel & extracted curve | Grid px={dbg.get('grid_px')}",
        use_column_width=True)

    label, score, clean = predict_window(raw)
    
    st.subheader("Prediction")
    st.success(f"{label}\nConfidence: {score:.2f}")


    c1, c2 = st.columns([1,1])
    with c1:
        st.subheader("Prediction")
        st.success(f"{label}  \nConfidence: {score:.2f}")

        # Insights
        hr, rmssd, rpeaks = compute_hr_hrv(clean[:WINDOW], fs=FS)
        qrs_w = estimate_intervals(clean[:WINDOW], fs=FS, peaks=rpeaks)

        st.markdown("### Vital Insights (demo)")
        st.metric("Estimated Heart Rate (BPM)", f"{hr:.1f}" if not np.isnan(hr) else "N/A")
        st.metric("HRV (RMSSD, ms)", f"{rmssd:.1f}" if not np.isnan(rmssd) else "N/A")
        st.metric("Approx. QRS Width (s)", f"{qrs_w:.3f}" if not np.isnan(qrs_w) else "N/A")

        st.info("**Note:** Values are approximations for educational demo; not for clinical use.")

    with c2:
        st.subheader("Waveform (first 5 seconds)")
        fig, ax = plt.subplots(figsize=(10,3))
        seg = clean[:WINDOW]
        ax.plot(seg)
        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude (norm.)")
        ax.set_title("Filtered, normalized ECG")
        st.pyplot(fig)

    # Optional: show R-peaks overlay
    with st.expander("Show R-peaks overlay"):
        from ecg_utils import detect_rpeaks_simple
        peaks = detect_rpeaks_simple(seg, fs=FS)
        fig2, ax2 = plt.subplots(figsize=(10,3))
        ax2.plot(seg)
        ax2.scatter(peaks, seg[peaks], marker='x')
        ax2.set_title("Detected peaks (demo)")
        st.pyplot(fig2)

st.caption("© Demo for academic purposes only. Not a medical device.")



