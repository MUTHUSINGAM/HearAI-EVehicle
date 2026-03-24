import { useRef, useState } from "react";
import HearAITable from "./HearAITable";

const API_URL = "http://127.0.0.1:8000/predict";

export default function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [recording, setRecording] = useState(false);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);

  const submitAudio = async (file) => {
    setLoading(true);
    try {
      const form = new FormData();
      form.append("file", file, file.name || "recording.webm");
      const res = await fetch(API_URL, { method: "POST", body: form });
      const json = await res.json();
      setResult(json);
    } catch (e) {
      setResult({ error: String(e) });
    } finally {
      setLoading(false);
    }
  };

  const onUpload = async (e) => {
    const file = e.target.files?.[0];
    if (file) await submitAudio(file);
  };

  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new MediaRecorder(stream);
    chunksRef.current = [];
    recorder.ondataavailable = (ev) => chunksRef.current.push(ev.data);
    recorder.onstop = async () => {
      const blob = new Blob(chunksRef.current, { type: "audio/webm" });
      const file = new File([blob], "realtime.webm", { type: "audio/webm" });
      await submitAudio(file);
      stream.getTracks().forEach((t) => t.stop());
    };
    mediaRecorderRef.current = recorder;
    recorder.start();
    setRecording(true);
  };

  const stopRecording = () => {
    mediaRecorderRef.current?.stop();
    setRecording(false);
  };

  const statusTone = result?.error
    ? "error"
    : result?.predicted_class === "healthy"
      ? "ok"
      : "warn";

  return (
    <div className="page-bg">
      <div className="app-shell">
        <header className="hero">
          <div>
            <p className="eyebrow">Intelligent Acoustic Diagnostics</p>
            <h1>HearAI-EV Control Center</h1>
            <p className="sub">
              Upload EV audio or record live from microphone to detect bearing/propeller faults and get clear actions.
            </p>
          </div>
          <div className="hero-badges">
            <span className="badge">Real-time Ready</span>
            <span className="badge">LLM Explainability</span>
            <span className="badge">Edge AI</span>
          </div>
        </header>

        <section className="panel">
          <div className="panel-title-row">
            <h2>Audio Input</h2>
            <span className={recording ? "live-dot on" : "live-dot"}>{recording ? "Recording" : "Idle"}</span>
          </div>
          <div className="controls">
            <label className="file-input">
              <input type="file" accept="audio/*" onChange={onUpload} />
              <span>Upload Audio File</span>
            </label>
            {!recording ? (
              <button className="btn btn-primary" onClick={startRecording}>🎤 Start Mic</button>
            ) : (
              <button className="btn btn-danger" onClick={stopRecording}>⏹ Stop Mic</button>
            )}
          </div>
        </section>

        <section className={`panel result-panel ${statusTone}`}>
          <div className="panel-title-row">
            <h2>Diagnosis Result</h2>
            {loading && <span className="loader">Analyzing audio...</span>}
          </div>
          {!result && !loading && (
            <p className="muted">No prediction yet. Upload a file or start microphone capture.</p>
          )}
          {result && (
            <>
              {result.error ? (
                <div className="result-error">{result.error}</div>
              ) : (
                <div className="result-grid">
                  <div className="metric-card">
                    <div className="metric-label">Predicted Class</div>
                    <div className="metric-value">{result.predicted_class}</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-label">Confidence</div>
                    <div className="metric-value">{(result.confidence * 100).toFixed(2)}%</div>
                  </div>
                  <div className="metric-card grow">
                    <div className="metric-label">Explanation</div>
                    <div className="metric-value small">{result.diagnostic?.message}</div>
                  </div>
                  <div className="metric-card grow">
                    <div className="metric-label">Recommended Action</div>
                    <div className="metric-value small">{result.diagnostic?.recommended_action}</div>
                  </div>
                </div>
              )}
            </>
          )}
        </section>

        <section className="panel">
          <div className="panel-title-row">
            <h2>Fault Reference Matrix</h2>
            <span className="muted">Based on your defined fault feature categories</span>
          </div>
          <HearAITable />
        </section>
      </div>
    </div>
  );
}
