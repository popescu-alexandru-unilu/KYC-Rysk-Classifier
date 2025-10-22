import os
from pathlib import Path

def _fail(msg):
    raise RuntimeError(msg)

def load_model():
    path = Path(os.getenv("MODEL_CKPT", ""))  # file OR directory
    if not path.exists():
        _fail(f"MODEL_CKPT not found: {path}")

    # Hugging Face directory?
    if path.is_dir():
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            from .model_minilm import MiniLMClassifier
            m = MiniLMClassifier()
            m.model = AutoModelForSequenceClassification.from_pretrained(str(path))
            m.tokenizer = AutoTokenizer.from_pretrained(str(path))
            return {"type":"hf", "model": m}
        except Exception as e:
            _fail(f"HF load failed for dir {path}: {e}")

    # Single file: choose by extension (fallback for old checkpoints)
    ext = path.suffix.lower()
    try:
        if ext in (".pt", ".pth"):
            from .infer_minilm import load_model as load_old
            m = load_old(str(path), "cpu")
            return {"type":"torch", "model": m}
    except Exception as e:
        _fail(f"Model load failed for {path}: {e}")

    _fail(f"Unsupported MODEL_CKPT: {path} (ext: {ext})")
