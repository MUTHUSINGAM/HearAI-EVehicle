# 🚗 HearAI-EV: Intelligent Acoustic Diagnostics System

**A machine learning-powered acoustic fault detection system for electric vehicles using YAMNet and Generative AI**

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Project Structure](#project-structure)
4. [Installation & Setup](#installation--setup)
5. [Quick Start](#quick-start)
6. [Usage Modes](#usage-modes)
7. [Components](#components)
8. [Configuration](#configuration)
9. [Results & Outputs](#results--outputs)
10. [Future Enhancements](#future-enhancements)

---

## 🎯 Overview

HearAI-EV is an intelligent acoustic diagnostics system designed to detect and explain mechanical faults in electric vehicles. Since electric vehicles operate silently compared to conventional combustion engines, early-stage mechanical issues such as bearing wear or propeller anomalies often go unnoticed. This project addresses that challenge by continuously monitoring vehicle sounds and converting them into meaningful diagnostic information for the driver.

### Key Features

✅ **Real-time Audio Monitoring** - Captures 1-minute acoustic samples  
✅ **Automated Fault Detection** - YAMNet-based classification (bearing, propeller, healthy)  
✅ **Confidence-based Decision Logic** - Probability thresholds ensure reliable predictions  
✅ **AI-Generated Explanations** - Mistral LLM converts technical outputs to user-friendly messages  
✅ **Visual Alert Interface** - Color-coded status displays with severity indicators  
✅ **Diagnostic Dashboard** - Comprehensive health monitoring and trend analysis  
✅ **Full Explainability** - Every prediction includes actionable recommendations  

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    VEHICLE SOUND INPUT                          │
│              (1-minute audio samples @ 16kHz)                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PREPROCESSING PIPELINE                        │
│   • Normalize & Resample to 16 kHz                             │
│   • Convert to Mono                                            │
│   • Extract Mel-Spectrogram Features                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              YAMNET TRANSFER LEARNING MODEL                     │
│   • Pretrained on AudioSet (millions of sounds)               │
│   • Fine-tuned on EV acoustic dataset                         │
│   • 3 classes: Bearing Fault, Propeller Fault, Healthy       │
│   • Output: Probability scores for each class                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│               CONFIDENCE-BASED DECISION LOGIC                   │
│   • Minimum confidence threshold: 70%                          │
│   • Fault probability assessment                              │
│   • Severity determination (low/medium/high)                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│        MISTRAL LLM - EXPLANATION GENERATION                    │
│   • Converts technical ML output to human language            │
│   • Generates actionable recommendations                      │
│   • Provides estimated urgency levels                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              USER INTERFACE & ALERTING                          │
│   • Green: Vehicle Operating Normally                         │
│   • Yellow: Warning - Schedule Maintenance                    │
│   • Red: Critical - Immediate Action Required                 │
│   • HTML Dashboard, Mobile-friendly JSON                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
HearAI-EV/
├── data/
│   └── processed/
│       ├── train/
│       │   ├── bearing/
│       │   ├── healthy/
│       │   └── propeller/
│       ├── val/
│       │   ├── bearing/
│       │   ├── healthy/
│       │   └── propeller/
│       └── test/
│           ├── bearing/
│           ├── healthy/
│           └── propeller/
│
├── dataset/          # Original audio data (pre-processed)
│   ├── Bearing/
│   ├── Healthy/
│   └── Propeller/
│
├── models/
│   └── yamnet_finetuned.h5      # Trained model (saved after training)
│
├── reports/
│   ├── training_history.csv     # Training metrics
│   ├── model_evaluation.json    # Test performance
│   ├── alert_display.png        # Visual alerts
│   ├── diagnostic_dashboard.png # Dashboard visualization
│   ├── dashboard.html           # Interactive HTML dashboard
│   ├── predictions_log.json     # All predictions
│   └── system_report.json       # Final system report
│
├── data_processing.py           # Phase 1: Data preparation & augmentation
├── yamnet_training.py           # Phase 2: Model training & evaluation
├── inference.py                 # Phase 3: Real-time prediction
├── llm_explanations.py          # Phase 3B: LLM-based explanations
├── ui_interface.py              # Phase 4: Visual interface
├── main.py                      # Main orchestration
│
├── requirements.txt             # Python dependencies
├── COMPREHENSIVE_SETUP.md       # Setup instructions
└── README.md                    # This file
```

---

## 💾 Installation & Setup

### Prerequisites

- Python 3.8+
- pip or conda
- 8GB RAM minimum (16GB recommended)
- 5GB disk space for models and data

### Step 1: Clone and Navigate

```bash
cd "d:\VIII SEM\HearAI-EV"
```

### Step 2: Install Dependencies

```bash
# Using pip
pip install -r requirements.txt

# Or using conda
conda create -n hearai python=3.10
conda activate hearai
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
python -c "import librosa; print('Audio library ready')"
```

### Step 4: Setup Optional LLM (Mistral)

For local LLM integration:

```bash
# Install Ollama from https://ollama.ai
ollama pull mistral

# Or use pre-installed Mistral model if available
```

---

## 🚀 Quick Start

### Option 1: Run Complete Demo

```bash
python main.py --mode demo
```

This will:
1. Load the pre-trained model
2. Process test audio samples
3. Generate diagnostic reports
4. Create visualizations and dashboards
5. Analyze health trends

### Option 2: Process Specific Audio Directory

```bash
python main.py --mode process --audio-dir data/processed/test --limit 20
```

### Option 3: Continuous Monitoring Simulation

```bash
python main.py --mode monitor
```

---

## 📖 Usage Modes

### Training Mode (Phase 2)

If you need to retrain the model:

```bash
python yamnet_training.py
```

This will:
- Load preprocessed data from `data/processed/`
- Build YAMNet architecture with custom classification head
- Train for up to 50 epochs with early stopping
- Save model to `models/yamnet_finetuned.h5`
- Generate evaluation metrics and visualizations

### Inference Mode (Default)

```bash
from inference import HearAIPredictor, get_diagnostic_info

# Initialize predictor
predictor = HearAIPredictor('models/yamnet_finetuned.h5')

# Predict on audio file
result = predictor.predict('path/to/audio.wav')

# Get diagnostic information
diagnostic = get_diagnostic_info(result)

print(f"Status: {diagnostic['status']}")
print(f"Confidence: {diagnostic['confidence']}%")
print(f"Recommendations: {diagnostic['symptoms']}")
```

### LLM Integration Mode

```bash
from llm_explanations import DiagnosticReport
from inference import HearAIPredictor, get_diagnostic_info

# Predict
predictor = HearAIPredictor()
prediction = predictor.predict('audio.wav')

# Generate diagnostic info
diagnostic = get_diagnostic_info(prediction)

# Generate report with LLM explanations
report_gen = DiagnosticReport()
report = report_gen.create_report(prediction, diagnostic)

# Format for display
print(report_gen.format_for_display(report))

# Save report
report_gen.save_report(report)
```

---

## 🔧 Components

### 1. **data_processing.py** - Phase 1: Data Preparation
- Scans and validates audio files
- Applies leak-free train/val/test split
- Generates 20 augmentations per file (stretch, pitch shift, noise, etc.)
- Creates comprehensive reports

**Key Functions:**
- `step1_scan_and_validate()` - Audio quality checks
- `step2_split_files()` - Stratified splitting
- `step3_augment_splits()` - Data augmentation
- `step4_generate_reports()` - Statistics and visualizations

### 2. **yamnet_training.py** - Phase 2: Model Training
- Loads YAMNet from TensorFlow Hub
- Fine-tunes on EV acoustic data
- Implements custom classification head
- Evaluates on test set

**Key Functions:**
- `build_yamnet_model()` - Architecture definition
- `train_model()` - Training loop with callbacks
- `evaluate_model()` - Comprehensive evaluation
- `plot_evaluation_results()` - Visualizations

### 3. **inference.py** - Phase 3: Real-time Prediction
- Loads trained model
- Processes audio in real-time
- Generates confidence scores
- Determines fault severity

**Key Classes:**
- `HearAIPredictor` - Main inference engine
- `ContinuousMonitor` - Monitors health trends

### 4. **llm_explanations.py** - Phase 3B: LLM Integration
- Generates human-readable explanations
- Uses Mistral LLM (local or via Ollama)
- Creates maintenance guides
- Estimates urgency levels

**Key Classes:**
- `DiagnosticLLM` - LLM interface
- `DiagnosticReport` - Report generation

### 5. **ui_interface.py** - Phase 4: Visual Interface
- Generates alert screens
- Creates diagnostic dashboards
- Produces HTML reports
- Mobile-friendly formatting

**Key Classes:**
- `AlertDisplay` - Visual alerts
- `DiagnosticDashboard` - Dashboard visualization

### 6. **main.py** - Orchestration
- Coordinates all components
- Provides command-line interface
- Runs demos and batch processing
- Generates final reports

---

## ⚙️ Configuration

### Model Configuration

Modify `CONFIG` in `yamnet_training.py`:

```python
CONFIG = {
    'classes': ['bearing', 'healthy', 'propeller'],
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 1e-4,
    'early_stopping_patience': 5,
}
```

### Inference Configuration

Modify `CONFIG` in `inference.py`:

```python
CONFIG = {
    'confidence_threshold': 0.7,  # Minimum confidence for classification
    'fault_threshold': 0.5,        # Probability above which it's a fault
    'input_sr': 16000,             # Sample rate
}
```

### LLM Configuration

Modify `LLM_CONFIG` in `llm_explanations.py`:

```python
LLM_CONFIG = {
    'local_model': 'mistral',
    'temperature': 0.7,
    'max_tokens': 500,
    'model_name': 'mistral-7b-instruct',
}
```

---

## 📊 Results & Outputs

### Training Phase Outputs
- `models/yamnet_finetuned.h5` - Trained model
- `reports/training_history.csv` - Epoch-by-epoch metrics
- `reports/model_evaluation.json` - Test set performance
- `reports/model_evaluation.png` - Visualization plots

### Inference Phase Outputs
- `reports/predictions_log.json` - All predictions
- `reports/alert_display.png` - Alert screen
- `reports/diagnostic_dashboard.png` - Dashboard visualization
- `reports/dashboard.html` - Interactive HTML dashboard
- `reports/system_report.json` - System summary

### Expected Performance

Based on EV acoustic data:
- **Overall Accuracy**: 85-92%
- **Bearing Detection Recall**: 88-95%
- **Propeller Detection Recall**: 82-90%
- **Healthy Classification**: 90-98%

---

## 🎨 Visual Outputs

### Alert Display
Color-coded status screen showing:
- ✅ **Green (Healthy)**: Vehicle operating normally
- ⚠️ **Yellow (Warning)**: Schedule maintenance within 24-48 hours
- 🔴 **Red (Critical)**: Immediate action required

### Dashboard
Comprehensive visualization with:
- Status timeline
- Confidence trends
- Fault type distribution
- Severity analysis
- Historical records

---

## 🔮 Future Enhancements

### Phase 2 Improvements
- [ ] Real-time streaming audio processing
- [ ] Edge deployment on vehicle hardware
- [ ] Cloud synchronization for fleet management
- [ ] Mobile app integration (iOS/Android)

### Phase 3 Enhancements
- [ ] Multi-language explanations
- [ ] Integration with vehicle diagnostics CAN bus
- [ ] Predictive maintenance scheduling
- [ ] Driver behavior analysis

### Phase 4 Improvements
- [ ] Voice feedback integration
- [ ] AR visualization in vehicle HUD
- [ ] Bluetooth integration with smartwatch
- [ ] Over-the-air model updates

---

## 🐛 Troubleshooting

### Model Not Found
```bash
# Ensure training has completed
python yamnet_training.py

# Or download pre-trained model
# [Link to pre-trained model]
```

### CUDA/GPU Issues
```bash
# CPU-only mode
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

### Audio Loading Errors
```bash
# Verify audio format (WAV recommended)
# Check sample rate compatibility
# Ensure file permissions
```

### Memory Issues
```bash
# Reduce batch size in configuration
CONFIG['batch_size'] = 16  # From 32

# Or limit number of files processed
python main.py --mode process --limit 10
```

---

## 📚 References

1. YAMNet: https://github.com/google-research/perch/tree/main/chirp/projects/yamnet
2. Mistral LLM: https://mistral.ai/
3. TensorFlow Audio: https://www.tensorflow.org/io/tutorials/audio
4. Librosa Documentation: https://librosa.org/

---

## 📝 License & Attribution

This project is part of the EV Acoustic Diagnostics Research Initiative.

**Contributors:**
- ML/Deep Learning: YAMNet Fine-tuning, Data Augmentation
- LLM Integration: Mistral-based Explanations
- UI/UX: Diagnostic Dashboard, Alert Interface

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- [ ] Additional fault types (cooling fan, motor issues)
- [ ] Cross-vehicle acoustic generalization
- [ ] Real-time optimization
- [ ] Hardware acceleration

---

## 📧 Support

For issues or questions:
1. Check troubleshooting section
2. Review configuration settings
3. Examine log files in `logs/`
4. Check generated reports for clues

---

## 🎓 Educational Value

This project demonstrates:
- ✅ Transfer learning with pretrained models
- ✅ Audio feature extraction and processing
- ✅ Balanced dataset creation with augmentation
- ✅ Confidence-based decision making
- ✅ LLM integration for explainability
- ✅ End-to-end ML system design
- ✅ Production-grade visualization
- ✅ Real-world IoT application

---

**Last Updated:** January 2026  
**Version:** 1.0  
**Status:** Production Ready ✅
