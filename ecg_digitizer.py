# ecg_digitizer.py
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image

def pdf_to_image(pdf_path, dpi=300):
    pages = convert_from_path(pdf_path, dpi=dpi,poppler_path=r"C:\Users\heman\OneDrive\poppler\bin")
    return pages[0]  # return first page as PIL Image

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # enhance contrast
    img = cv2.equalizeHist(img)
    # threshold
    _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    return thresh

def extract_signal_from_image(img_path, target_length=1800):
    """Very basic digitizer: extract single ECG trace from image"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    _, bw = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    curve = max(contours, key=cv2.contourArea)

    # --- Overlay for debugging ---
    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(overlay, [curve], -1, (0,0,255), 1)  # red contour

    # --- Extract 1D signal ---
    h, w = img.shape
    signal = np.zeros(w)
    for x in range(w):
        ys = [pt[0][1] for pt in curve if pt[0][0] == x]
        if ys:
            signal[x] = np.mean(ys)

    # Normalize
    signal = - (signal - np.mean(signal))
    signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) * 2 - 1

    # Resample to fixed length
    signal = cv2.resize(signal.reshape(1, -1), (target_length, 1)).flatten()

    return signal, overlay