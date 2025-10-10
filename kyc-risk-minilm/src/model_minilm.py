# src/model_minilm.py
# MiniLM-L6 encoder + masked-mean pooling + linear head
# ------------------------------------------------------
# CHANGE: set NUM_LABELS to 3 (ternary) or 2 (binary).
# CHANGE: only if you swap models, update MODEL_NAME and HIDDEN_SIZE.

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"  # MiniLM-L6
HIDDEN_SIZE = 384                                       # MiniLM-L6 hidden dim
NUM_LABELS  = 3                                         # <<< CHANGE to 2 if you built binary labels
MAX_LEN     = 512

class MiniLMClassifier(nn.Module):
    """
    MiniLM-L6 text classifier.
    Input:  list[str] texts (evidence strings)
    Output: logits tensor [B, NUM_LABELS]
    """
    def __init__(self):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.enc = AutoModel.from_pretrained(MODEL_NAME)  # returns last_hidden_state
        self.head = nn.Linear(HIDDEN_SIZE, NUM_LABELS)

    def _masked_mean(self, hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        hidden:    [B, T, D]
        attn_mask: [B, T] with 1 for real tokens, 0 for padding
        returns:   [B, D]
        """
        mask = attn_mask.unsqueeze(-1)                  # [B, T, 1]
        summed = (hidden * mask).sum(dim=1)             # [B, D]
        denom  = mask.sum(dim=1).clamp(min=1)           # [B, 1]
        return summed / denom

    def featurize(self, texts, device="cpu", max_len=MAX_LEN) -> torch.Tensor:
        """
        Tokenize + encode texts to pooled embeddings. Returns [B, HIDDEN_SIZE].
        """
        batch = self.tok(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )
        batch = {k: v.to(device) for k, v in batch.items()}

        out = self.enc(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        last_hidden = out.last_hidden_state             # [B, T, D]
        pooled = self._masked_mean(last_hidden, batch["attention_mask"])  # [B, D]
        return pooled

    def forward(self, texts, device="cpu"):
        """
        Return unnormalized logits [B, NUM_LABELS].
        Use .softmax(-1) on the result to get probabilities.
        """
        emb = self.featurize(texts, device=device)      # [B, D]
        logits = self.head(emb)                         # [B, C]
        return logits
    
# --- SMOKE TEST (runs only if you execute this file directly) ---
if __name__ == "__main__":
    m = MiniLMClassifier()
    txt = "[KYC] Name: CASPRO TECHNOLOGY LTD. [COUNTRY] Hong Kong [SANCTIONS] list=US-BIS-EL"
    with torch.no_grad():
        probs = m([txt]).softmax(-1)[0].tolist()
    print("probs:", probs)


