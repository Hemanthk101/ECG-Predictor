# ECG-Predictor
❤️ ECG Health Predictor  An end-to-end AI application that analyzes ECG data — either as raw signal files or scanned ECG images — to predict cardiac health conditions such as arrhythmias and compute key vitals like heart rate, HRV, and QRS width.


Key Features

🧠 Deep Learning Model (TensorFlow/Keras)
Detects normal vs. abnormal (arrhythmic) ECG signals using a CNN-LSTM architecture trained on MIT-BIH data.

🩺 ECG Signal Analytics
Computes vital parameters — heart rate, heart rate variability (RMSSD), and QRS interval width.

🖼️ Image-to-Signal Digitization (OpenCV + U-Net)
Extracts ECG waveforms from scanned PDF or image reports, removes gridlines and annotations, and reconstructs a clean signal.

🌐 Interactive Web App (Streamlit / FastAPI + React)
Upload an ECG file, view extracted waveform, predictions, and downloadable reports — all in real time.

☁️ Cloud-ready Deployment
Containerized for deployment on AWS / GCP / HuggingFace Spaces.

🧩 Tech Stack
Layer	Technologies
Frontend	Streamlit (current) → migrating to React/Next.js
Backend / API	FastAPI, Python
Modeling	TensorFlow, Keras, Scikit-learn
Signal Processing	NumPy, SciPy
Computer Vision	OpenCV, pdf2image, U-Net segmentation
Deployment	Docker, AWS EC2, HuggingFace Spaces
Data	MIT-BIH Arrhythmia Dataset, PTB-XL ECG Database
📊 Workflow

Upload ECG File
Supports .csv, .png, .jpg, or .pdf.

Digitization (if scanned)

Detect ECG lead panels (Lead II preferred).

Remove gridlines and text using OpenCV morphology.

Convert waveform pixels → time-amplitude signal.

Signal Pre-processing

Bandpass filtering, z-score normalization, windowing.

Model Prediction

CNN-LSTM model outputs probability of arrhythmia.

Visualization

Display waveform plot, contour overlay, and key vitals.

Report Generation (coming soon)

Export results as a PDF summary for clinicians/researchers.

🧠 Model Architecture
Input (360Hz windowed ECG)
        ↓
Conv1D + ReLU
        ↓
LSTM (sequence learning)
        ↓
Dense Layer + Sigmoid
        ↓
Binary Output (Normal / Abnormal)

📈 Future Improvements

Train custom U-Net digitizer for multi-lead ECG panels.

Expand classification beyond binary (e.g., AFib, PVC, LBBB).

Integrate transformer-based ECG encoders (ECG-BERT).

Add FHIR-compatible API for clinical interoperability.

Develop a modern FastAPI + React production dashboard.

⚙️ How to Run
# 1. Clone repo
git clone https://github.com/<yourusername>/ecg-health-predictor.git
cd ecg-health-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Streamlit app
streamlit run app.py

📄 Disclaimer

This project is intended for educational and research purposes only.
It is not a certified medical device and should not be used for real clinical diagnosis.

🤝 Contributing

Pull requests and ideas are welcome!
If you’d like to collaborate on improving the digitization pipeline or model accuracy, feel free to open an issue or fork the repo.
