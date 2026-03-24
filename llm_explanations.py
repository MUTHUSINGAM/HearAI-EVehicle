# Stub: Generates LLM-based human-readable explanations (call to local LLM, e.g. Mistral, or fallback)
def generate_explanation(pred_class, confidence, all_probs):
    if pred_class == 'healthy':
        return "The vehicle sounds normal. No issues detected at this time. Continue safe operation."
    msg = f"A potential {pred_class} fault was detected with confidence {confidence:.2f}. "
    if pred_class == 'bearing':
        msg += "Bearings may be experiencing early-stage wear. It is recommended to schedule a maintenance check soon."
    elif pred_class == 'propeller':
        msg += "Unusual propeller noise detected. Please consult service personnel at your earliest convenience."
    msg += "\nAll probabilities: " + ", ".join([f"{k}: {v:.2f}" for k, v in all_probs.items()])
    return msg

# For real system, connect to local LLM model here:
# from llama_cpp import Llama
# llm = Llama(model_path="./your-mistral-model.bin")
# def generate_explanation(...): ...