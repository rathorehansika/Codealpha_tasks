"""Shared audio preprocessing & feature extraction.

IMPORTANT: train.py and main.py BOTH import from here so that the exact same
preprocessing is applied at training time and at inference time. A mismatch
between the two is the classic cause of a model that scores well on the test
set but produces nonsense on real microphone audio.
"""
import numpy as np
import librosa

SR = 22050          # target sample rate
N_MFCC = 40
N_FEATURES = 182    # 40 mfcc + 12 chroma + 128 mel + 1 zcr + 1 rms


def preprocess(y, sr):
    """Trim leading/trailing silence and peak-normalize amplitude.

    This makes a quiet laptop-mic clip and a loud studio clip look comparable
    to the model, which is the single biggest win for real-world robustness.
    """
    # Trim silence (top_db: anything 25 dB below peak is treated as silence)
    y, _ = librosa.effects.trim(y, top_db=25)
    if y.size == 0:
        return y
    # Peak normalize to [-1, 1]
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return y


def extract_features(y, sr):
    """Return the 182-dim feature vector used by the model."""
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=y).T, axis=0)
    return np.hstack((mfccs, chroma, mel, zcr, rms))


def load_and_features(path):
    """Load an audio file, preprocess, and extract features. Returns None on failure/empty."""
    y, sr = librosa.load(path, sr=SR)
    y = preprocess(y, sr)
    if y.size < SR // 2:  # shorter than ~0.5s after trimming -> unusable
        return None
    return extract_features(y, sr)


def augmentations(y, sr):
    """Yield augmented copies of a (already preprocessed) signal.

    Used ONLY on training data to teach the model real-world variation:
    background noise, pitch differences, and speaking-rate differences.
    """
    augs = []

    # 1. Additive white noise (simulates mic / room noise)
    noise = np.random.normal(0, 0.005, size=y.shape)
    augs.append(np.clip(y + noise, -1.0, 1.0))

    # 2. Pitch shift up and down (different voices / intonation)
    for steps in (2, -2):
        try:
            augs.append(librosa.effects.pitch_shift(y=y, sr=sr, n_steps=steps))
        except Exception:
            pass

    # 3. Time stretch slower / faster (different speaking rates)
    for rate in (0.9, 1.1):
        try:
            augs.append(librosa.effects.time_stretch(y=y, rate=rate))
        except Exception:
            pass

    return augs
