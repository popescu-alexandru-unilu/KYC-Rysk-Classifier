# src/model_minilm.py
# MiniLM-L6 sequence classifier using Hugging Face AutoModelForSequenceClassification
# ------------------------------------------------------
# CHANGE: set NUM_LABELS to 3 (ternary) or 2 (binary).
# CHANGE: only if you swap models, update MODEL_NAME.

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"  # MiniLM-L6
NUM_LABELS  = 3                                         # <<< CHANGE to 2 if you built binary labels
MAX_LEN     = 512

class MiniLMClassifier(nn.Module):
    """
    MiniLM-L6 text classifier using HF AutoModelForSequenceClassification.
    Input:  list[str] texts (evidence strings)
    Output: logits tensor [B, NUM_LABELS]
    """
    def __init__(self):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def featurize(self, texts, device="cpu", max_len=MAX_LEN) -> torch.Tensor:
        """
        Tokenize + encode texts to pooled embeddings (for compatibility).
        Returns [B, HIDDEN_SIZE].
        """
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        # Use the model's pooling (HF handles pooling automatically in forward)
        with torch.no_grad():
            # Use the encoder (base model) to produce hidden states or pooler output
            outputs = self.model.base_model(**batch)
            pooled = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state.mean(dim=1)
        return pooled

    def forward(self, texts, device="cpu"):
        """
        Return unnormalized logits [B, NUM_LABELS].
        Use .logits in output for probabilities.
        """
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        self.to(device)
        # Call the HF model to obtain logits
        outputs = self.model(**batch)
        return outputs.logits

# --- SMOKE TEST (runs only if you execute this file directly) ---
if __name__ == "__main__":
    m = MiniLMClassifier()
    txt = "[KYC] Name: CASPRO TECHNOLOGY LTD. [COUNTRY] Hong Kong [SANCTIONS] list=US-BIS-EL"
    with torch.no_grad():
        logits = m([txt])
        probs = torch.softmax(logits, dim=-1)[0].tolist()
    print("probs:", probs)
