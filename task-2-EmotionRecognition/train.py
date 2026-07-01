import os
import glob
import time
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, BatchNormalization, MaxPooling1D, Flatten, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from features import preprocess, extract_features, augmentations, SR, N_FEATURES

# Configurations
DATA_DIR = "data"
MODEL_PATH = "emotion_model.keras"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"

EMOTIONS_MAP = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgusted",
    "08": "surprised"
}


def parse_files():
    """STEP 1: collect (file_path, label) pairs from the RAVDESS dataset."""
    print("STEP 1: Dataset setup - parsing files and labels...")
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory '{DATA_DIR}' not found. Download dataset first.")

    file_paths = glob.glob(os.path.join(DATA_DIR, "**", "*.wav"), recursive=True)
    if len(file_paths) == 0:
        raise FileNotFoundError(f"No .wav files found in '{DATA_DIR}'.")

    items = []
    for path in file_paths:
        parts = os.path.basename(path).split('-')
        if len(parts) < 7:
            continue
        emotion_code = parts[2]
        if emotion_code not in EMOTIONS_MAP:   # skip 'calm' (02) and anything unmapped
            continue
        items.append((path, EMOTIONS_MAP[emotion_code]))

    print(f"Found {len(items)} usable files across {len(EMOTIONS_MAP)} emotions.")
    return items


def build_feature_set(items, augment=False):
    """STEP 2: load + preprocess + extract features for a list of (path, label).

    When augment=True (training split only), each clip also contributes several
    augmented variants. Test data is never augmented to keep the score honest.
    """
    X, y = [], []
    skipped = 0
    start = time.time()
    for idx, (path, label) in enumerate(items):
        try:
            sig, sr = librosa.load(path, sr=SR)
            sig = preprocess(sig, sr)
            if sig.size < SR // 2:
                skipped += 1
                continue

            X.append(extract_features(sig, sr)); y.append(label)

            if augment:
                for aug in augmentations(sig, sr):
                    X.append(extract_features(aug, sr)); y.append(label)
        except Exception as e:
            print(f"  ! error on {os.path.basename(path)}: {e}")
            skipped += 1

        if (idx + 1) % 100 == 0:
            print(f"  processed {idx + 1}/{len(items)} files...  (samples so far: {len(X)})")

    print(f"  -> {len(X)} feature vectors in {time.time() - start:.1f}s (skipped {skipped}).")
    return np.array(X), np.array(y)


def build_lstm_model(input_shape):
    return Sequential([
        Input(shape=input_shape),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(7, activation='softmax')
    ])


def build_cnn_model(input_shape):
    return Sequential([
        Input(shape=input_shape),
        Conv1D(64, 3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(128, 3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(7, activation='softmax')
    ])


def train_one(name, build_fn, ckpt_path, X_tr, y_tr_cat, callbacks):
    print(f"\n--- Training {name} ---")
    if name == "LSTM":
        X_tr_model = X_tr.reshape((X_tr.shape[0], 1, N_FEATURES))
        model = build_fn((1, N_FEATURES))
    else:
        X_tr_model = X_tr.reshape((X_tr.shape[0], N_FEATURES, 1))
        model = build_fn((N_FEATURES, 1))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    t0 = time.time()
    history = model.fit(X_tr_model, y_tr_cat, validation_split=0.1, epochs=100,
                        batch_size=32, callbacks=callbacks, verbose=1)
    print(f"{name} finished in {time.time() - t0:.1f}s.")
    if os.path.exists(ckpt_path):
        model = tf.keras.models.load_model(ckpt_path)
    return model, history


def plot_history(history, title, fname):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'{title} Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{title} Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.tight_layout(); plt.savefig(fname); plt.close()


def plot_cm(y_true, y_pred, names, title, fname, cmap):
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d',
                xticklabels=names, yticklabels=names, cmap=cmap)
    plt.title(title); plt.ylabel('True'); plt.xlabel('Predicted')
    plt.tight_layout(); plt.savefig(fname); plt.close()


def main():
    items = parse_files()
    paths = [p for p, _ in items]
    labels = [l for _, l in items]

    # STEP 3: split FIRST (on files) so augmented clips never leak into the test set
    train_items, test_items = train_test_split(
        items, test_size=0.20, random_state=42, stratify=labels
    )
    print(f"\nSplit: {len(train_items)} train files, {len(test_items)} test files.")

    # Load from scratch cache if available to speed up run
    if os.path.exists("scratch/X_train.npy") and os.path.exists("scratch/X_test.npy"):
        print("Loading pre-extracted features from cache...")
        X_train = np.load("scratch/X_train.npy")
        y_train_str = np.load("scratch/y_train_str.npy")
        X_test = np.load("scratch/X_test.npy")
        y_test_str = np.load("scratch/y_test_str.npy")
    else:
        print("\nSTEP 2a: extracting TRAIN features (with augmentation)...")
        X_train, y_train_str = build_feature_set(train_items, augment=True)
        print("STEP 2b: extracting TEST features (no augmentation)...")
        X_test, y_test_str = build_feature_set(test_items, augment=False)

    # STEP 3: encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    joblib.dump(label_encoder, ENCODER_PATH)
    y_train = label_encoder.transform(y_train_str)
    y_test = label_encoder.transform(y_test_str)
    print(f"\nLabel encoder saved -> {ENCODER_PATH}. Classes: {list(label_encoder.classes_)}")

    # Standardize (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved -> {SCALER_PATH}")

    # STEP 4: balance classes with SMOTE (on the augmented training set)
    print("\nSTEP 4: balancing classes with SMOTE...")
    X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)
    print(f"Training set after SMOTE: {X_train.shape}")

    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=7)

    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=1e-6)

    print("\nSTEP 5: training models...")
    lstm_model, lstm_hist = train_one(
        "LSTM", build_lstm_model, "lstm_best.keras", X_train, y_train_cat,
        [early_stop, reduce_lr, ModelCheckpoint("lstm_best.keras", monitor='val_accuracy', save_best_only=True)])
    cnn_model, cnn_hist = train_one(
        "1D CNN", build_cnn_model, "cnn_best.keras", X_train, y_train_cat,
        [EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True), reduce_lr,
         ModelCheckpoint("cnn_best.keras", monitor='val_accuracy', save_best_only=True)])

    # STEP 6: evaluate
    print("\nSTEP 6: evaluation...")
    X_test_lstm = X_test.reshape((X_test.shape[0], 1, N_FEATURES))
    X_test_cnn = X_test.reshape((X_test.shape[0], N_FEATURES, 1))
    lstm_preds = np.argmax(lstm_model.predict(X_test_lstm), axis=1)
    cnn_preds = np.argmax(cnn_model.predict(X_test_cnn), axis=1)
    lstm_acc = accuracy_score(y_test, lstm_preds)
    cnn_acc = accuracy_score(y_test, cnn_preds)

    names = label_encoder.classes_
    print("\n" + "=" * 50)
    print(f"LSTM  Test Accuracy: {lstm_acc * 100:.2f}%")
    print(f"CNN   Test Accuracy: {cnn_acc * 100:.2f}%")
    print("=" * 50)
    print("\nLSTM report:\n", classification_report(y_test, lstm_preds, target_names=names))
    print("\nCNN report:\n", classification_report(y_test, cnn_preds, target_names=names))

    os.makedirs("plots", exist_ok=True)
    plot_history(lstm_hist, "LSTM", "plots/lstm_learning_curves.png")
    plot_history(cnn_hist, "1D CNN", "plots/cnn_learning_curves.png")
    plot_cm(y_test, lstm_preds, names, "LSTM Confusion Matrix", "plots/lstm_confusion_matrix.png", "Blues")
    plot_cm(y_test, cnn_preds, names, "1D CNN Confusion Matrix", "plots/cnn_confusion_matrix.png", "Reds")
    print("Saved plots to 'plots/'.")

    # STEP 7: save the better model
    if lstm_acc >= cnn_acc:
        lstm_model.save(MODEL_PATH); best = ("LSTM", lstm_acc)
    else:
        cnn_model.save(MODEL_PATH); best = ("CNN", cnn_acc)
    print(f"\nBest model: {best[0]} ({best[1] * 100:.2f}%) saved -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
