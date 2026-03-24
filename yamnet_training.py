from pathlib import Path

import joblib
import numpy as np
try:
    import tensorflow_hub as hub
    _TF_AVAILABLE = True
except Exception:
    hub = None
    _TF_AVAILABLE = False

event_labels = ["bearing", "propeller", "healthy"]

_yamnet_model = None
_classifier = None


def load_model():
    """Load YAMNet + optional EV classifier, with safe fallback."""
    global _yamnet_model, _classifier
    if _TF_AVAILABLE and _yamnet_model is None:
        _yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    if _classifier is None:
        classifier_path = Path("models/ev_classifier.pkl")
        if classifier_path.exists():
            _classifier = joblib.load(classifier_path)
    if _yamnet_model is not None:
        return {"mode": "yamnet", "model": _yamnet_model}
    return {"mode": "fallback", "model": None}


def _fallback_classify(x: np.ndarray):
    """
    TensorFlow-free fallback classifier.
    Uses simple spectral ratio heuristics to keep app usable when TF DLLs fail.
    """
    # Windowed FFT power spectrum
    if x.size == 0:
        probs = np.array([0.2, 0.2, 0.6], dtype=np.float32)
        return "healthy", float(probs[2]), {k: float(v) for k, v in zip(event_labels, probs)}
    window = np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x * window)) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1.0 / 16000.0)

    total = float(np.sum(spec) + 1e-8)
    low = float(np.sum(spec[(freqs >= 20) & (freqs < 500)])) / total
    mid = float(np.sum(spec[(freqs >= 500) & (freqs < 3000)])) / total
    high = float(np.sum(spec[(freqs >= 3000) & (freqs <= 8000)])) / total
    rms = float(np.sqrt(np.mean(x ** 2)))

    # Heuristic mapping:
    # - Propeller tends to stronger high band tonal/noisy components
    # - Bearing tends to stronger mid-band roughness
    # - Healthy tends to lower RMS + smoother spectrum
    bearing_score = 0.55 * mid + 0.2 * low + 0.25 * min(rms * 5.0, 1.0)
    propeller_score = 0.55 * high + 0.15 * mid + 0.30 * min(rms * 5.0, 1.0)
    healthy_score = max(0.0, 1.0 - (0.75 * (bearing_score + propeller_score)))

    probs = np.array([bearing_score, propeller_score, healthy_score], dtype=np.float32)
    probs = np.clip(probs, 1e-6, None)
    probs = probs / np.sum(probs)
    pred_idx = int(np.argmax(probs))
    return event_labels[pred_idx], float(probs[pred_idx]), {
        k: float(v) for k, v in zip(event_labels, probs)
    }


def classify_audio(model, waveform):
    """Classify a mono 16kHz waveform."""
    global _classifier
    x = waveform.astype(np.float32)
    if np.max(np.abs(x)) > 0:
        x = x / np.max(np.abs(x))
    if isinstance(model, dict) and model.get("mode") == "fallback":
        return _fallback_classify(x)

    raw_model = model["model"] if isinstance(model, dict) else model
    scores, embeddings, _ = raw_model(x)
    if _classifier is not None:
        emb = np.mean(embeddings.numpy(), axis=0).reshape(1, -1)
        pred_probs = _classifier.predict_proba(emb)[0]
        pred_idx = int(np.argmax(pred_probs))
        return event_labels[pred_idx], float(pred_probs[pred_idx]), {
            k: float(v) for k, v in zip(event_labels, pred_probs)
        }
    avg_scores = np.mean(scores.numpy(), axis=0)
    # In-range fallback proxies for demos only.
    class_inds = [318, 479, 0]
    probs = avg_scores[class_inds]
    probs = probs / (np.sum(probs) + 1e-8)
    pred_idx = int(np.argmax(probs))
    return event_labels[pred_idx], float(probs[pred_idx]), {
        k: float(v) for k, v in zip(event_labels, probs)
    }
