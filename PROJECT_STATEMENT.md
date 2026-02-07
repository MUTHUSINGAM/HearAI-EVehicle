# HearAI-EV: Intelligent Acoustic Diagnostics for Electric Vehicles

## Project Overview

**HearAI-EV** is an intelligent acoustic diagnostics system designed to detect and explain mechanical faults in electric vehicles using a combination of machine learning and Generative AI. Unlike conventional combustion engines that produce audible engine noise, electric vehicles operate silently, making early-stage mechanical issues such as bearing wear or propeller anomalies difficult to detect. This project addresses this critical challenge by continuously monitoring vehicle sounds and converting them into meaningful, actionable diagnostic information for drivers.

---

## Core Concept

### The Problem
Electric vehicles lack the audible feedback mechanisms of traditional engines. Mechanical faults like:
- **Bearing wear** (ball/roller degradation)
- **Propeller damage** (aerodynamic anomalies)
- **Other mechanical issues**

...often go unnoticed until they become severe, leading to:
- Unexpected breakdowns
- Expensive repairs
- Safety risks
- Reduced vehicle lifespan

### The Solution
HearAI-EV transforms silent EV operation into an intelligent monitoring system that:
1. **Continuously captures** vehicle sounds during normal driving
2. **Processes** audio in real-time using advanced ML models
3. **Detects** faults with high accuracy
4. **Explains** findings in plain language using Generative AI
5. **Alerts** drivers with clear, actionable recommendations

---

## System Architecture

### 1. **Real-Time Audio Acquisition**
- Microphone sensors installed in the vehicle
- Captures sound in **1-minute intervals** (efficient processing)
- Standardized format: **16 kHz, mono, float32 PCM**

### 2. **Preprocessing Pipeline**
- **Resampling** to 16 kHz (standard audio rate)
- **Mono conversion** (stereo → single channel)
- **Feature extraction** (MFCCs, log-mel spectrograms)
- **Noise robustness** (handles environmental variations)

### 3. **Machine Learning Classification**
- **Base Model**: YAMNet (pretrained on millions of real-world sounds)
- **Transfer Learning**: Fine-tuned on custom EV acoustic dataset
- **Classes**: 
  - `bearing` - Bearing wear/fault
  - `propeller` - Propeller damage/anomaly
  - `healthy` - Normal operation
- **Output**: Probability scores for each class

### 4. **Fault Detection Logic**
- **Confidence-based decision**: Thresholds determine fault vs. normal
- **Severity assessment**: High/Medium/Low based on confidence scores
- **Temporal analysis**: Tracks trends over time for predictive maintenance

### 5. **Generative AI Explanation (LLM Integration)**
- **Local LLM**: Mistral (or similar) running on-device
- **Purpose**: Converts technical ML output into human-readable explanations
- **Output includes**:
  - Nature of the issue (what's wrong)
  - Severity level (how serious)
  - Recommended actions (what to do)
  - Maintenance urgency (when to act)

### 6. **Visual Alert Interface**
- **Green indicator**: Vehicle operating normally
- **Warning symbol**: Fault detected
- **Clear messages**: LLM-generated explanations
- **Actionable guidance**: Next steps for the driver

---

## Technology Stack

### Machine Learning
- **TensorFlow**: Deep learning framework
- **YAMNet**: Pretrained audio classification model (TensorFlow Hub)
- **Transfer Learning**: Fine-tuning on EV-specific dataset
- **Feature Engineering**: Librosa for audio processing

### Generative AI / LLM
- **Mistral LLM**: Local, on-device language model
- **Purpose**: Natural language explanation generation
- **Privacy**: All processing happens locally (no cloud dependency)

### User Interface
- **Streamlit**: Web-based dashboard
- **Real-time updates**: Live diagnostic feedback
- **Mobile-friendly**: Responsive design for in-vehicle use

### Audio Processing
- **Librosa**: Audio feature extraction
- **SoundFile**: Audio I/O
- **NumPy/SciPy**: Signal processing

---

## Key Innovations

### 1. **Hybrid Edge-AI Architecture**
- ML model runs locally on the vehicle
- No internet connection required for inference
- Fast, real-time responses
- Privacy-preserving (data stays on-device)

### 2. **Transfer Learning Approach**
- Leverages YAMNet's pretrained knowledge (millions of sounds)
- Fine-tuned on small, domain-specific EV dataset
- Reduces data requirements and training time
- Improves accuracy on EV-specific sounds

### 3. **LLM-Enhanced Explainability**
- Traditional ML: "Class: bearing, Confidence: 0.85"
- HearAI-EV: "A potential bearing fault was detected with 85% confidence. Bearings may be experiencing early-stage wear. It is recommended to schedule a maintenance check soon."
- Makes diagnostics accessible to non-technical users

### 4. **Real-Time Processing**
- 1-minute audio chunks processed immediately
- Low latency (< 2 seconds from capture to alert)
- Continuous monitoring during normal driving

---

## Workflow Example

```
1. Vehicle is driving normally
   ↓
2. Microphone captures 1-minute audio sample
   ↓
3. Audio preprocessed (resample, mono, features extracted)
   ↓
4. YAMNet model classifies: [bearing: 0.15, propeller: 0.05, healthy: 0.80]
   ↓
5. Decision Logic: Confidence > 0.8 → "healthy"
   ↓
6. LLM generates: "The vehicle sounds normal. No issues detected."
   ↓
7. UI displays: 🟢 "Vehicle Operating Normally"
```

**Fault Scenario:**
```
1. Vehicle is driving with bearing wear
   ↓
2. Microphone captures audio with abnormal sounds
   ↓
3. Audio preprocessed
   ↓
4. YAMNet classifies: [bearing: 0.85, propeller: 0.10, healthy: 0.05]
   ↓
5. Decision Logic: bearing > 0.7 → "FAULT DETECTED"
   ↓
6. LLM generates: "A potential bearing fault was detected with 85% confidence. 
   Bearings may be experiencing early-stage wear. Schedule maintenance soon."
   ↓
7. UI displays: ⚠️ "Fault Detected" + explanation
```

---

## Benefits

### For Drivers
- **Early fault detection**: Catch issues before they become critical
- **Clear explanations**: Understand what's wrong in plain language
- **Actionable guidance**: Know exactly what to do
- **Safety**: Prevent unexpected breakdowns

### For Maintenance Teams
- **Predictive maintenance**: Schedule repairs proactively
- **Detailed diagnostics**: Understand fault severity
- **Cost savings**: Fix issues early (cheaper repairs)
- **Data-driven**: Evidence-based maintenance decisions

### For Vehicle Manufacturers
- **Quality assurance**: Monitor vehicle health in real-world conditions
- **Warranty optimization**: Identify issues before warranty claims
- **Customer satisfaction**: Proactive support
- **Data insights**: Learn from real-world usage patterns

---

## Future Enhancements

1. **Multi-sensor fusion**: Combine audio with vibration, temperature sensors
2. **Cloud analytics**: Aggregate data across vehicle fleet
3. **Predictive models**: Forecast failures weeks in advance
4. **Integration**: Connect with vehicle telematics systems
5. **Mobile app**: Standalone smartphone application
6. **Advanced LLM**: More sophisticated explanation generation

---

## Technical Approach Summary

**Machine Learning**: Transfer learning with YAMNet → Fine-tuned on EV dataset → 3-class classification

**Generative AI**: Local Mistral LLM → Converts ML output → Human-readable explanations

**Architecture**: Edge-AI (on-device) → Real-time processing → Low latency → Privacy-preserving

**User Experience**: Visual alerts → Clear messages → Actionable recommendations → Non-technical friendly

---

## Conclusion

HearAI-EV demonstrates how **Machine Learning** can provide accurate acoustic fault detection, while **Generative AI** enhances explainability and user interaction. By combining real-time audio monitoring, transfer learning, and local LLM-based explanations, the system provides a practical, scalable, and intelligent solution for modern electric vehicle diagnostics.

The system transforms silent EV operation into an intelligent, proactive maintenance tool that keeps vehicles safe, reliable, and cost-effective to maintain.
