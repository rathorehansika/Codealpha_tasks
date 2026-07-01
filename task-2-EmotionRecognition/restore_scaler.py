"""Regenerate the ORIGINAL scaler.pkl and label_encoder.pkl that match the
existing emotion_model.keras. This exactly replays the original training-time
feature extraction (no trimming/normalization, no augmentation) and the same
stratified 80/20 split (random_state=42), then refits the StandardScaler on the
training portion -- producing artifacts identical to the originals.
"""
import os
import glob
import time
import librosa
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_DIR = "data"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"

EMOTIONS_MAP = {
    "01": "neutral", "03": "happy", "04": "sad", "05": "angry",
    "06": "fearful", "07": "disgusted", "08": "surprised"
}


def main():
    file_paths = glob.glob(os.path.join(DATA_DIR, "**", "*.wav"), recursive=True)
    print(f"Found {len(file_paths)} wav files. Extracting original features...")

    features_list, labels_list = [], []
    t0 = time.time()
    for idx, fp in enumerate(file_paths):
        parts = os.path.basename(fp).split('-')
        if len(parts) < 7:
            continue
        code = parts[2]
        if code not in EMOTIONS_MAP:
            continue
        try:
            y, sr = librosa.load(fp, sr=22050)
            mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
            chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
            mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
            zcr = np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)
            rms = np.mean(librosa.feature.rms(y=y).T, axis=0)
            features_list.append(np.hstack((mfccs, chroma, mel, zcr, rms)))
            labels_list.append(EMOTIONS_MAP[code])
        except Exception as e:
            print(f"  skip {os.path.basename(fp)}: {e}")
        if (idx + 1) % 200 == 0:
            print(f"  {idx + 1}/{len(file_paths)}...", flush=True)

    X = np.array(features_list)
    y = np.array(labels_list)
    print(f"Extracted {X.shape} in {time.time() - t0:.1f}s.")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    joblib.dump(label_encoder, ENCODER_PATH)
    print(f"Saved {ENCODER_PATH}. Classes: {list(label_encoder.classes_)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )
    scaler = StandardScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved {SCALER_PATH}. Restore complete.")


if __name__ == "__main__":
    main()
