# Task 2 - Emotion Recognition from Speech

Recognize human emotions (e.g., happy, angry, sad) from speech audio using deep learning and speech signal processing techniques. Built as part of the CodeAlpha ML internship.

## Objective
Recognize human emotions from speech audio using deep learning.

## Approach
- Extracted MFCC (Mel-Frequency Cepstral Coefficients) features from raw audio
- Trained and compared CNN and LSTM architectures for emotion classification
- Selected the best-performing model based on validation accuracy

## Models
- **CNN** (`cnn_best.keras`) — trained on extracted audio features
- **LSTM** (`lstm_best.keras`) — trained on sequential audio features
- **Best overall model**: `emotion_model.keras`

## Project Structure
task-2-EmotionRecognition/
├── plots/                  # Confusion matrices & learning curves
├── cnn_best.keras          # Trained CNN model
├── lstm_best.keras         # Trained LSTM model
├── emotion_model.keras     # Final selected model
├── features.py             # Audio feature extraction (MFCCs)
├── train.py                # Model training script
├── main.py                 # Main script to run predictions
├── restore_scaler.py       # Loads/restores the feature scaler
├── download_dataset.py     # Script to download the dataset
├── scaler.pkl              # Fitted scaler for feature normalization
├── label_encoder.pkl       # Encoder for emotion class labels
├── requirements.txt        # Python dependencies
└── train_log                # Training run logs
## Dataset
Not included in this repo due to size (2.83 GB).
Dataset used: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
