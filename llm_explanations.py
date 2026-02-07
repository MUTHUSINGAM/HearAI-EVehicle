"""
Fast LLM-based explanation generator for fault detection
Ensures < 2 second latency with human-readable, non-technical descriptions
"""
import time
import os
from pathlib import Path

# Global LLM instance (lazy loaded)
_llm = None
_llm_available = False

def _get_available_ollama_model():
    """Get the best available Ollama model (prioritize small, fast models)"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=1)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m.get('name', '') for m in models]
            
            # Priority order: smallest/fastest first for < 2s latency
            preferred_models = [
                "qwen2.5:1.5b",      # Smallest, newest, fastest
                "llama3.2:1b",        # Very small, fast
                "phi3.5:latest",     # Small, good quality
                "llama3.2:3b",        # Small-medium
                "mistral:latest",     # Medium, good quality
                "llama3:latest",       # Larger but good
            ]
            
            # Find first available model
            for preferred in preferred_models:
                if preferred in model_names:
                    return preferred
            
            # Fallback to any available model
            if model_names:
                return model_names[0]
    except:
        pass
    return None

def _init_llm():
    """Initialize local LLM if available"""
    global _llm, _llm_available
    
    if _llm is not None:
        return _llm_available
    
    # Try Ollama first (fastest, easiest setup)
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=1)
        if response.status_code == 200:
            model_name = _get_available_ollama_model()
            if model_name:
                _llm_available = True
                _llm = "ollama"
                print(f"✓ Using Ollama model: {model_name} for LLM explanations")
                return True
    except ImportError:
        pass  # requests not installed
    except Exception:
        pass  # Ollama not running or connection failed
    
    # Try llama-cpp-python (quantized models)
    try:
        from llama_cpp import Llama
        model_paths = [
            "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            "models/phi-2.Q4_K_M.gguf",
            "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                _llm = Llama(
                    model_path=model_path,
                    n_ctx=512,  # Small context for speed
                    n_threads=4,
                    verbose=False
                )
                _llm_available = True
                print(f"✓ Using {model_path} for LLM explanations")
                return True
    except ImportError:
        pass
    except Exception as e:
        print(f"⚠️  LLM model found but failed to load: {e}")
    
    _llm_available = False
    return False
    
def _generate_with_ollama(pred_class, confidence, severity):
    """Generate explanation using Ollama API"""
    try:
        try:
            import requests
        except ImportError:
            return None
        
        # Get best available model
        model_name = _get_available_ollama_model()
        if not model_name:
            return None
        
        # Determine severity level
        if confidence >= 0.85:
            sev_text = "high"
            urgency = "soon"
        elif confidence >= 0.70:
            sev_text = "moderate"
            urgency = "within a few days"
        else:
            sev_text = "low"
            urgency = "when convenient"
        
        # Create prompt
        if pred_class == 'bearing':
            issue = "bearing wear or damage"
            context = "The bearings in your vehicle may be wearing out. This is like the wheels of a bicycle getting wobbly - they need attention."
        elif pred_class == 'propeller':
            issue = "propeller damage or imbalance"
            context = "Your vehicle's propeller (the spinning part that moves air) may have damage or be out of balance."
        else:
            return None
        
        prompt = f"""Write a brief, friendly explanation for a vehicle owner about {issue} detected in their electric vehicle.

Confidence: {int(confidence*100)}%
Severity: {sev_text}
Urgency: {urgency}

Requirements:
- Use simple, everyday language (no technical jargon)
- Be reassuring but clear about the issue
- Suggest what to do next
- Keep it under 3 sentences
- Sound like a helpful mechanic talking to a friend

Context: {context}

Write the explanation:"""
        
        # Adjust timeout based on model size
        timeout = 1.2 if "1b" in model_name or "1.5b" in model_name else 1.8
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "max_tokens": 100,  # Short response for speed
                    "top_p": 0.9
                }
            },
            timeout=timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            explanation = result.get('response', '').strip()
            # Clean up the response - take first meaningful sentence/paragraph
            lines = explanation.split('\n')
            explanation = lines[0].strip()
            # Remove any prefix like "Explanation:" or "Response:"
            for prefix in ["Explanation:", "Response:", "Here's", "Here is"]:
                if explanation.startswith(prefix):
                    explanation = explanation[len(prefix):].strip()
            return explanation
        
    except Exception as e:
        print(f"Ollama error: {e}")
    
    return None

def _generate_with_llama_cpp(pred_class, confidence, severity):
    """Generate explanation using llama-cpp-python"""
    global _llm
    
    if _llm is None:
        return None
    
    try:
        # Determine severity
        if confidence >= 0.85:
            sev_text = "high"
            urgency = "soon"
        elif confidence >= 0.70:
            sev_text = "moderate"
            urgency = "within a few days"
        else:
            sev_text = "low"
            urgency = "when convenient"
        
        if pred_class == 'bearing':
            issue = "bearing wear"
            context = "bearings wearing out like bicycle wheels getting wobbly"
        elif pred_class == 'propeller':
            issue = "propeller damage"
            context = "propeller (spinning part) may be damaged or unbalanced"
        else:
            return None
        
        prompt = f"""Explain {issue} in simple words. Confidence: {int(confidence*100)}%. Severity: {sev_text}. Urgency: {urgency}. Context: {context}. Keep it short and friendly, like talking to a friend. What should they do?"""
        
        start_time = time.time()
        response = _llm(
            prompt,
            max_tokens=80,  # Short for speed
            temperature=0.7,
            stop=["\n\n", "Note:", "Technical:"],
            echo=False
        )
        elapsed = time.time() - start_time
        
        if elapsed < 1.8:  # Ensure under 2 seconds
            explanation = response['choices'][0]['text'].strip()
            return explanation
        
    except Exception as e:
        print(f"LLM error: {e}")
    
    return None

def _generate_template_explanation(pred_class, confidence, all_probs):
    """Fallback: Generate human-readable explanation from templates"""
    
    # Determine severity and urgency
    if confidence >= 0.85:
        severity = "high"
        urgency = "soon"
        urgency_text = "It's a good idea to get this checked soon"
    elif confidence >= 0.70:
        severity = "moderate"
        urgency = "moderate"
        urgency_text = "You should plan to have this looked at within a few days"
    else:
        severity = "low"
        urgency = "low"
        urgency_text = "Keep an eye on this and schedule a check when convenient"
    
    # Generate friendly, non-technical explanations
    if pred_class == 'healthy':
        return "✅ Your vehicle sounds normal! Everything is working as expected. No action needed - you're good to keep driving."
    
    elif pred_class == 'bearing':
        conf_pct = int(confidence * 100)
        
        if severity == "high":
            return f"⚠️ We detected unusual sounds from your vehicle's bearings (the parts that help things spin smoothly). " \
                   f"The system is {conf_pct}% confident about this. Think of it like a bicycle wheel getting wobbly - " \
                   f"it still works, but it needs attention. {urgency_text} to prevent bigger problems later."
        
        elif severity == "moderate":
            return f"⚠️ Your vehicle's bearings might be starting to wear out. The system detected this with {conf_pct}% confidence. " \
                   f"This is like early warning signs - nothing urgent, but worth checking. {urgency_text}."
        
        else:
            return f"ℹ️ We noticed some subtle changes in your vehicle's sound that might indicate early bearing wear. " \
                   f"Confidence is {conf_pct}%. This is very minor - just something to keep in mind. " \
                   f"You can mention it during your next regular maintenance visit."
    
    elif pred_class == 'propeller':
        conf_pct = int(confidence * 100)
        
        if severity == "high":
            return f"⚠️ We detected unusual sounds from your vehicle's propeller (the spinning part that moves air). " \
                   f"The system is {conf_pct}% confident about this. It might be damaged or out of balance. " \
                   f"{urgency_text} to avoid performance issues."
        
        elif severity == "moderate":
            return f"⚠️ Your vehicle's propeller may have some damage or imbalance. Detected with {conf_pct}% confidence. " \
                   f"This could affect how well your vehicle moves air. {urgency_text}."
        
        else:
            return f"ℹ️ We noticed some minor changes in propeller sounds. Confidence is {conf_pct}%. " \
                   f"This is very minor - just something to keep an eye on. Mention it during your next service visit."
    
    # Fallback
    return f"We detected a potential {pred_class} issue with {int(confidence*100)}% confidence. " \
           f"{urgency_text} to have it checked out."

def generate_explanation(pred_class, confidence, all_probs):
    """
    Generate human-readable, non-technical explanation for fault detection.
    Ensures < 2 second latency using fast LLM or smart templates.
        
        Args:
        pred_class: 'bearing', 'propeller', or 'healthy'
        confidence: Confidence score (0-1)
        all_probs: Dictionary of all class probabilities
        
        Returns:
        Human-readable explanation string
    """
    start_time = time.time()
    
    # Healthy case - quick return
    if pred_class == 'healthy':
        return "✅ Your vehicle sounds normal! Everything is working as expected. No action needed - you're good to keep driving."
    
    # Determine severity
    if confidence >= 0.85:
        severity = "high"
    elif confidence >= 0.70:
        severity = "moderate"
    else:
        severity = "low"
    
    # Try LLM first (if available and fast)
    _init_llm()
    
    if _llm_available:
        explanation = None
        
        if _llm == "ollama":
            explanation = _generate_with_ollama(pred_class, confidence, severity)
        elif _llm is not None:
            explanation = _generate_with_llama_cpp(pred_class, confidence, severity)
        
        # Use LLM result if we got it quickly
        if explanation and (time.time() - start_time) < 1.8:
            return explanation
    
    # Fallback to smart templates (always fast, always human-readable)
    explanation = _generate_template_explanation(pred_class, confidence, all_probs)
    
    elapsed = time.time() - start_time
    if elapsed > 0.1:  # Only log if it took noticeable time
        print(f"Explanation generated in {elapsed:.2f}s")
    
    return explanation