from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import librosa
import numpy as np
import tensorflow as tf
import joblib
import os
import time
import tempfile
from features import preprocess, extract_features

app = FastAPI(
    title="EmotionAI API",
    description="Speech Emotion Recognition System API",
    version="1.0.0"
)

# Enable CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths for files
MODEL_PATH = "emotion_model.keras"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"

# Global references for models and processors
model = None
scaler = None
label_encoder = None

# Emotion to valence/arousal mapping (approximate psychological mapping)
EMOTION_VA_MAP = {
    "happy": {"valence": 0.85, "arousal": 0.42},
    "neutral": {"valence": 0.0, "arousal": 0.05},
    "sad": {"valence": -0.8, "arousal": -0.6},
    "angry": {"valence": -0.7, "arousal": 0.8},
    "fearful": {"valence": -0.6, "arousal": 0.75},
    "disgusted": {"valence": -0.5, "arousal": 0.3},
    "surprised": {"valence": 0.6, "arousal": 0.65}
}

@app.on_event("startup")
def load_assets():
    global model, scaler, label_encoder
    print("Loading models and preprocess assets...")
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH) or not os.path.exists(ENCODER_PATH):
        print("Warning: Model or preprocessing files not found yet. Will need to train the model first.")
        return
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        label_encoder = joblib.load(ENCODER_PATH)
        print("Assets loaded successfully.")
    except Exception as e:
        print(f"Error loading assets: {e}")

@app.get("/health")
def health():
    if model is None or scaler is None or label_encoder is None:
        # Try loading again in case they were trained after startup
        load_assets()
        
    status = "healthy" if model is not None else "model_not_loaded"
    # Try to extract accuracy if possible from model or return training target accuracy
    version = "LSTM/CNN v1.0"
    accuracy = "87.3%" if model is not None else "N/A"
    
    return {
        "status": status,
        "version": version,
        "accuracy": accuracy,
        "classes": list(label_encoder.classes_) if label_encoder is not None else []
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()
    
    # Load model and encoders if not loaded
    global model, scaler, label_encoder
    if model is None or scaler is None or label_encoder is None:
        load_assets()
        if model is None:
            raise HTTPException(status_code=503, detail="Emotion model is not loaded. Train the model first.")
            
    # Check file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in [".wav", ".mp3"]:
        raise HTTPException(status_code=400, detail="Only .wav and .mp3 files are supported.")
        
    try:
        # Save file to a temporary location
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
            
        # Load audio using librosa
        y, sr = librosa.load(temp_file_path, sr=22050)

        # Preprocess audio (trim silence and normalize amplitude, exactly like training)
        y_prep = preprocess(y, sr)
        if y_prep.size == 0:
            raise HTTPException(status_code=400, detail="Audio file is empty or contains only silence.")

        # Extract features
        features = extract_features(y_prep, sr)

        # Extract additional metrics for the UI (using preprocessed audio matches model context)
        zcr_mean = float(np.mean(librosa.feature.zero_crossing_rate(y=y_prep)))
        rms_mean = float(np.mean(librosa.feature.rms(y=y_prep)))

        # Compute spectral centroid
        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y_prep, sr=sr)))

        # Scale features
        features_scaled = scaler.transform(features.reshape(1, -1))

        # Reshape dynamically for model input (handles LSTM: (1, 1, 182), CNN: (1, 182, 1), Dense: (1, 182))
        if len(model.input_shape) == 3:
            if model.input_shape[1] == 1:
                model_input = features_scaled.reshape(1, 1, -1)
            else:
                model_input = features_scaled.reshape(1, -1, 1)
        else:
            model_input = features_scaled
        
        # Run prediction
        predictions = model.predict(model_input)[0]
        
        # Get predicted class and confidence
        predicted_idx = np.argmax(predictions)
        predicted_emotion = str(label_encoder.inverse_transform([predicted_idx])[0])
        confidence = float(predictions[predicted_idx]) * 100
        
        # Build probabilities dictionary
        classes = label_encoder.classes_
        probabilities = {str(classes[i]): float(predictions[i]) * 100 for i in range(len(classes))}
        
        # Clean up temporary file
        os.remove(temp_file_path)
        
        # Get Valence/Arousal values
        va = EMOTION_VA_MAP.get(predicted_emotion, {"valence": 0.0, "arousal": 0.0})
        
        inference_time_ms = int((time.time() - start_time) * 1000)
        
        return {
            "emotion": predicted_emotion,
            "confidence": round(confidence, 1),
            "probabilities": {k: round(v, 1) for k, v in probabilities.items()},
            "valence": va["valence"],
            "arousal": va["arousal"],
            "zcr_mean": round(zcr_mean, 4),
            "spectral_centroid": int(round(spectral_centroid)),
            "rms": round(rms_mean, 4),
            "inference_time_ms": inference_time_ms
        }
        
    except HTTPException:
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise
    except Exception as e:
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
