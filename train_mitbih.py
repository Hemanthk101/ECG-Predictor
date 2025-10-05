# train_mitbih.py
import wfdb
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from ecg_utils import bandpass_filter, zscore, make_windows

# ---- Config ----
FS = 360
WINDOW = 1800        # 5 seconds @ 360 Hz
STEP = 1800          # non-overlapping
RECORDS = ["100", "101", "102", "103", "104"]  # expand this list for better results
PN_DIR = "mitdb"

def label_windows(ann_samples, idx_start, idx_end):
    # mark “abnormal” if any annotation symbol is not normal 'N' in window
    # (simplified for demo; refine using actual labels)
    return 1 if np.any((ann_samples >= idx_start) & (ann_samples < idx_end)) else 0

def build_model(input_len=WINDOW):
    model = Sequential([
        Conv1D(64, kernel_size=7, activation='relu', input_shape=(input_len, 1)),
        MaxPooling1D(pool_size=2),
        Conv1D(64, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        LSTM(64, return_sequences=False),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    X, y = [], []
    for rec in RECORDS:
        record = wfdb.rdrecord(rec, pn_dir=PN_DIR)
        ann = wfdb.rdann(rec, 'atr', pn_dir=PN_DIR)

        sig = record.p_signal[:, 0].astype(float)
        sig = bandpass_filter(sig, fs=FS, low=0.5, high=40)
        sig = zscore(sig)

        windows = make_windows(sig, WINDOW, STEP)
        for wi in range(len(windows)):
            start = wi * STEP
            end = start + WINDOW
            label = label_windows(ann.sample, start, end)
            X.append(windows[wi])
            y.append(label)

    X = np.asarray(X).reshape(-1, WINDOW, 1)
    y = np.asarray(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model(WINDOW)
    model.summary()
    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=10,
        batch_size=32,
        verbose=1
    )

    # Evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {acc:.3f}")

    # Save
    import os
    os.makedirs("models", exist_ok=True)
    model.save("models/ecg_model.h5")
    print("Saved models/ecg_model.h5")

if __name__ == "__main__":
    main()
