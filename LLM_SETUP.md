# LLM Setup Guide for Fast Explanations

The HearAI-EV system generates human-readable, non-technical fault descriptions using LLMs. The system is designed to complete explanations within **2 seconds**.

## How It Works

The system uses a **three-tier approach** for fast, reliable explanations:

1. **LLM (if available)**: Uses local LLM for natural language generation
2. **Smart Templates (fallback)**: Pre-written, human-readable templates that always work
3. **Automatic fallback**: If LLM is slow or unavailable, uses templates instantly

## Option 1: Ollama (Recommended - Easiest Setup)

Ollama is the easiest way to get fast LLM explanations.

### Installation

1. **Download Ollama**: https://ollama.ai
2. **Install** the application
3. **Pull a small, fast model**:
   ```bash
   ollama pull llama3.2:1b
   ```
   Or for better quality (slightly slower):
   ```bash
   ollama pull mistral:7b
   ```

### Usage

Just run Ollama in the background - the system will automatically detect and use it!

```bash
# Start Ollama (runs in background)
ollama serve
```

The system will automatically connect to `http://localhost:11434` when generating explanations.

**Advantages:**
- ✅ Easiest setup
- ✅ Fast inference (< 1 second)
- ✅ Multiple model options
- ✅ Automatic model management

---

## Option 2: llama-cpp-python (Advanced)

For more control, use quantized models with llama-cpp-python.

### Installation

```bash
pip install llama-cpp-python
```

### Download a Quantized Model

Choose a small, fast model (recommended for < 2 second latency):

1. **TinyLlama** (fastest, ~1.1B parameters):
   ```bash
   # Download from HuggingFace
   wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
   mkdir -p models
   mv tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf models/
   ```

2. **Phi-2** (good quality, ~2.7B parameters):
   ```bash
   wget https://huggingface.co/microsoft/phi-2-gguf/resolve/main/phi-2.Q4_K_M.gguf
   mkdir -p models
   mv phi-2.Q4_K_M.gguf models/
   ```

3. **Mistral-7B** (best quality, slower):
   ```bash
   wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf
   mkdir -p models
   mv mistral-7b-instruct-v0.2.Q4_K_M.gguf models/
   ```

The system will automatically detect and use the first available model.

**Advantages:**
- ✅ Full control over model
- ✅ No external dependencies
- ✅ Works offline completely

**Disadvantages:**
- ⚠️ Requires manual model download
- ⚠️ Larger models may exceed 2-second latency

---

## Option 3: Smart Templates (No Setup Required)

**This is the default!** The system includes intelligent, human-readable templates that:
- ✅ Always work (no setup needed)
- ✅ Complete instantly (< 0.01 seconds)
- ✅ Use simple, non-technical language
- ✅ Provide actionable recommendations

The templates automatically adjust based on:
- Fault type (bearing, propeller, healthy)
- Confidence level (high, moderate, low)
- Severity assessment

**Example template output:**
> "⚠️ We detected unusual sounds from your vehicle's bearings (the parts that help things spin smoothly). The system is 87% confident about this. Think of it like a bicycle wheel getting wobbly - it still works, but it needs attention. It's a good idea to get this checked soon to prevent bigger problems later."

---

## Performance Targets

| Method | Latency | Quality | Setup Difficulty |
|--------|---------|---------|------------------|
| **Smart Templates** | < 0.01s | Good | None (default) |
| **Ollama (llama3.2:1b)** | ~0.5-1s | Very Good | Easy |
| **Ollama (mistral:7b)** | ~1-2s | Excellent | Easy |
| **llama-cpp (TinyLlama)** | ~0.8-1.5s | Very Good | Medium |
| **llama-cpp (Phi-2)** | ~1-2s | Excellent | Medium |

**Recommendation**: Start with **Smart Templates** (works immediately). If you want more natural language variation, add **Ollama** with a small model.

---

## Testing

To test if LLM is working:

```python
from llm_explanations import generate_explanation

# Test explanation
expl = generate_explanation('bearing', 0.87, {'bearing': 0.87, 'propeller': 0.08, 'healthy': 0.05})
print(expl)
```

You should see either:
- An LLM-generated explanation (if Ollama/llama-cpp is available)
- A smart template explanation (always works)

Both are human-readable and non-technical!

---

## Troubleshooting

### "LLM not available" messages
- This is normal! The system falls back to smart templates automatically
- Templates are fast, human-readable, and always work

### Ollama connection errors
- Make sure Ollama is running: `ollama serve`
- Check if it's on port 11434: `curl http://localhost:11434/api/tags`

### llama-cpp model not found
- Download a model to the `models/` directory
- Check the file name matches one of the expected paths in `llm_explanations.py`

### Explanations taking > 2 seconds
- Use a smaller model (TinyLlama or llama3.2:1b)
- The system will automatically fall back to templates if LLM is too slow

---

## Customization

To customize explanations, edit the `_generate_template_explanation()` function in `llm_explanations.py`. The templates are designed to be:
- Simple and non-technical
- Reassuring but clear
- Actionable (tells user what to do)
- Friendly and conversational
