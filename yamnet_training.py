import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

# Classes for EV context
event_labels = ['bearing', 'propeller', 'healthy']


def load_model():
    '''Download and load YAMNet (pretrained or locally fine-tuned).'''
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
    return yamnet_model


def classify_audio(model, features):
    '''Run YAMNet on features and return (class, confidence, all_probs dict).'''
    # Expand dims for batch
    patch = np.expand_dims(features, axis=0).astype(np.float32)
    # If input is log-mel spectrogram: YAMNet expects waveform (mono PCM at 16kHz, float32, -1..1). We'll assume input is waveform, not features.
    # TODO: adapt for direct waveform input or use audio preproc as YAMNet expects.
    # YAMNet actual usage expects mono waveform, not features:
    # So for now, just use the waveform pipeline: features arg here is actually waveform!
    scores, embeddings, spectrogram = model(patch)
    scores = scores.numpy()
    avg_scores = np.mean(scores, axis=0)
    # For this minimal demo, map YAMNet classes to our 3: Just pick arbitrary indices.
    class_inds = [11,187,550]  # Placeholder indices for mapping to ['bearing','propeller','healthy']
    probs = avg_scores[class_inds]
    pred_idx = np.argmax(probs)
    pred_class = event_labels[pred_idx]
    confidence = probs[pred_idx]
    all_probs = dict(zip(event_labels, probs))
    return pred_class, confidence, all_probs
