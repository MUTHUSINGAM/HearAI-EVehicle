import time


FAULT_REFERENCES = {
    "healthy": [
        {
            "class": "Healthy - Excellent",
            "nextService": "Routine check at 10,000 km or 12 months",
            "notes": "All systems nominal. No action needed.",
        }
    ],
    "bearing": [
        {
            "class": "Bearing - Outer Race",
            "signalFeature": "Periodic impulse spikes at BPFO",
            "severity": "Moderate -> High",
            "nextService": "Replace bearing within 200 km",
        },
        {
            "class": "Bearing - Lubrication Failure",
            "signalFeature": "Broadband noise rise",
            "severity": "High",
            "nextService": "Immediate relubrication. Inspect within 50 km.",
        },
    ],
    "propeller": [
        {
            "class": "Propeller - Blade Imbalance",
            "signalFeature": "Strong 1P rotational signature",
            "severity": "Moderate",
            "nextService": "Dynamic balancing within 300 km",
        },
        {
            "class": "Propeller - Blade Crack",
            "signalFeature": "Random spikes + harmonic distortion",
            "severity": "Critical",
            "nextService": "Immediate shutdown. Do not operate.",
        },
    ],
}


def _severity(confidence: float) -> str:
    if confidence >= 0.85:
        return "high"
    if confidence >= 0.7:
        return "moderate"
    return "low"


def generate_explanation(pred_class, confidence, all_probs):
    """Generate a user-friendly explanation in <2 seconds."""
    start = time.time()
    sev = _severity(confidence)
    if pred_class == "healthy":
        return {
            "message": "Your vehicle sounds normal. No issue is detected right now.",
            "severity": "none",
            "recommended_action": "Continue normal operation.",
            "reference": FAULT_REFERENCES["healthy"][0],
            "latency_ms": int((time.time() - start) * 1000),
        }
    top_ref = FAULT_REFERENCES[pred_class][0 if sev != "high" else -1]
    if pred_class == "bearing":
        text = (
            "We detected unusual bearing sound. This usually means wear is starting in "
            "the rotating support parts. It is safer to schedule service soon."
        )
    else:
        text = (
            "We detected unusual propeller/fan sound. This may indicate imbalance or blade damage. "
            "Please inspect it early to avoid bigger failure."
        )
    return {
        "message": text,
        "severity": sev,
        "recommended_action": top_ref["nextService"],
        "reference": top_ref,
        "probabilities": {k: round(float(v), 4) for k, v in all_probs.items()},
        "latency_ms": int((time.time() - start) * 1000),
    }