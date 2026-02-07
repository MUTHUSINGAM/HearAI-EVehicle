import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import joblib
from pathlib import Path
import os

# Classes for EV context
event_labels = ['bearing', 'propeller', 'healthy']

# Global variables for model caching
_yamnet_model = None
_classifier = None

def load_model():
    '''Load YAMNet and trained classifier.'''
    global _yamnet_model, _classifier
    
    # Load YAMNet
    if _yamnet_model is None:
        print("Loading YAMNet model...")
        _yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
    
    # Load trained classifier if available
    classifier_path = Path('models/ev_classifier.pkl')
    if _classifier is None:
        if classifier_path.exists():
            print("Loading trained EV classifier...")
            _classifier = joblib.load(classifier_path)
        else:
            print("⚠️  WARNING: No trained classifier found!")
            print("   Run 'python train_classifier.py' first to train on your dataset.")
            print("   Using fallback method (may be inaccurate).")
            _classifier = None
    
    return _yamnet_model


def classify_audio(model, features):
    '''Run YAMNet + classifier on waveform and return (class, confidence, all_probs dict).'''
    global _classifier
    
    # YAMNet expects mono waveform at 16kHz, float32, shape (N,)
    waveform = features.astype(np.float32)
    
    # Normalize waveform to [-1, 1]
    if np.max(np.abs(waveform)) > 0:
        waveform = waveform / np.max(np.abs(waveform))
    
    # Get YAMNet embeddings
    scores, embeddings, spectrogram = model(waveform)
    
    # Use trained classifier if available
    if _classifier is not None:
        # Average embeddings over time
        avg_embedding = np.mean(embeddings.numpy(), axis=0).reshape(1, -1)
        
        # Predict with trained classifier
        pred_probs = _classifier.predict_proba(avg_embedding)[0]
        pred_idx = np.argmax(pred_probs)
        pred_class = event_labels[pred_idx]
        confidence = pred_probs[pred_idx]
        
        # Create probability dict
        all_probs = {label: prob for label, prob in zip(event_labels, pred_probs)}
        
        return pred_class, confidence, all_probs
    else:
        # Fallback: Use arbitrary YAMNet class mappings (not accurate!)
        scores = scores.numpy()
        avg_scores = np.mean(scores, axis=0)
        # Use indices within YAMNet's 521-class output
        class_inds = [318, 479, 0]  # Engine, Propeller, Speech (demo only)
        probs = avg_scores[class_inds]
        # Normalize to probabilities
        probs = probs / (np.sum(probs) + 1e-8)
        pred_idx = np.argmax(probs)
        pred_class = event_labels[pred_idx]
        confidence = probs[pred_idx]
        all_probs = dict(zip(event_labels, probs))
        
        return pred_class, confidence, all_probs
