# Quick Start Guide - HearAI-EV

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Train the Classifier (IMPORTANT!)

Before using the system, you need to train a classifier on your EV dataset:

```bash
python train_classifier.py
```

This will:
- Extract YAMNet embeddings from all audio files in `data/processed/train/` and `data/processed/val/`
- Train a Random Forest classifier on these embeddings
- Save the trained model to `models/ev_classifier.pkl`

**Note**: This step is required for accurate classification. Without it, the system will use fallback mappings that are not accurate.

## Step 3: Run the Streamlit UI

```bash
streamlit run main.py
```

Then:
1. Upload a WAV file from your dataset
2. View the classification results
3. See LLM-generated explanations

## Troubleshooting

### "No trained classifier found" warning
- Run `python train_classifier.py` first
- Make sure your dataset is in `data/processed/train/` and `data/processed/val/`

### Incorrect classifications
- Ensure you've trained the classifier on your dataset
- Check that audio files are properly formatted (16kHz, mono, WAV)

### Import errors
- Run `pip install -r requirements.txt` again
- Check Python version (3.8+ recommended)
