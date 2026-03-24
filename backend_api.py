from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from data_processing import process_audio_file
from llm_explanations import generate_explanation
from yamnet_training import classify_audio, load_model

app = FastAPI(title="HearAI-EV API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    waveform = process_audio_file(tmp_path, target_sr=16000)
    pred_class, confidence, probs = classify_audio(model, waveform)
    explanation = generate_explanation(pred_class, confidence, probs)
    return {
        "predicted_class": pred_class,
        "confidence": round(float(confidence), 4),
        "probabilities": {k: round(float(v), 4) for k, v in probs.items()},
        "diagnostic": explanation,
    }
