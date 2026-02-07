import os
import sys
import streamlit as st
from data_processing import process_audio_file
from yamnet_training import load_model, classify_audio
from llm_explanations import generate_explanation

# Demo: Audio file upload (simulate real-time chunk)
st.set_page_config(page_title="HearAI-EV Diagnostic UI", layout="wide")
st.title("HearAI-EV: Acoustic Diagnostics")

uploaded_file = st.file_uploader("Upload 1-min vehicle audio (WAV, 16kHz or other)", type=["wav"])

if uploaded_file is not None:
    st.info("Processing audio...")
    features = process_audio_file(uploaded_file, target_sr=16000)
    model = load_model()
    pred_class, confidence, probs = classify_audio(model, features)
    st.write(f"Classification: {pred_class}")
    st.write(f"Confidence: {confidence:.2f}")

    if pred_class == 'healthy' and confidence > 0.8:
        st.success("🟢 Vehicle Operating Normally")
        expl = generate_explanation(pred_class, confidence, probs)
        st.markdown(f"### {expl}")
    else:
        st.warning("⚠️ Fault Detected")
        expl = generate_explanation(pred_class, confidence, probs)
        st.markdown(f"### {expl}")
        
        # Show technical details in expander (optional)
        with st.expander("📊 Technical Details"):
            st.write(f"**Classification:** {pred_class}")
            st.write(f"**Confidence:** {confidence:.2%}")
            st.write("**All Probabilities:**")
            for cls, prob in probs.items():
                st.write(f"  - {cls}: {prob:.2%}")
else:
    st.info("Upload a 1-min vehicle WAV to begin.")
