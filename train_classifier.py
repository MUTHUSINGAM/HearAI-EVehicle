"""
Train a classifier on YAMNet embeddings for EV fault detection
This creates a proper model trained on your dataset
"""
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = Path('data/processed')
CLASSES = ['bearing', 'propeller', 'healthy']
MODEL_DIR = Path('models')
MODEL_DIR.mkdir(exist_ok=True)

def extract_yamnet_embeddings(audio_path, yamnet_model):
    """Extract YAMNet embeddings from audio file"""
    import soundfile as sf
    import librosa
    
    # Load and preprocess audio
    y, sr = sf.read(audio_path)
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    
    # Normalize to [-1, 1]
    y = y.astype(np.float32)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    
    # Get YAMNet embeddings
    scores, embeddings, spectrogram = yamnet_model(y)
    # Average embeddings over time
    avg_embedding = np.mean(embeddings.numpy(), axis=0)
    return avg_embedding

def load_dataset():
    """Load all audio files from dataset and extract embeddings"""
    print("Loading YAMNet model...")
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
    
    X = []
    y = []
    
    print("Extracting embeddings from dataset...")
    for split in ['train', 'val']:
        split_path = DATA_DIR / split
        for class_name in CLASSES:
            class_path = split_path / class_name
            if not class_path.exists():
                continue
            
            files = list(class_path.glob('*.wav'))
            print(f"Processing {len(files)} files from {split}/{class_name}...")
            
            for i, audio_file in enumerate(files):
                try:
                    embedding = extract_yamnet_embeddings(audio_file, yamnet_model)
                    X.append(embedding)
                    y.append(class_name)
                    
                    if (i + 1) % 20 == 0:
                        print(f"  Processed {i+1}/{len(files)} files")
                except Exception as e:
                    print(f"  Error processing {audio_file}: {e}")
                    continue
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nDataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.bincount([CLASSES.index(c) for c in y])}")
    
    return X, y, yamnet_model

def train_classifier(X, y):
    """Train Random Forest classifier on embeddings"""
    print("\nTraining classifier...")
    
    # Encode labels
    label_to_idx = {label: idx for idx, label in enumerate(CLASSES)}
    y_encoded = np.array([label_to_idx[label] for label in y])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Train Random Forest
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASSES))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save model
    model_path = MODEL_DIR / 'ev_classifier.pkl'
    joblib.dump(clf, model_path)
    print(f"\nModel saved to {model_path}")
    
    return clf

if __name__ == '__main__':
    print("=" * 60)
    print("HearAI-EV: Training Classifier on YAMNet Embeddings")
    print("=" * 60)
    
    X, y, yamnet_model = load_dataset()
    clf = train_classifier(X, y)
    
    print("\n✅ Training complete!")
    print("You can now use the trained classifier in yamnet_training.py")
