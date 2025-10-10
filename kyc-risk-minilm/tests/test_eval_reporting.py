import json
from pathlib import Path


def test_confusion_matrix_script_smoke():
    # Use the existing eval script to compute a small confusion matrix
    # Import inside to avoid heavy deps on module import
    from api.src.eval_valid import MiniLMClassifier, predict_batch, confusion_matrix
    import torch

    ckpt_candidates = [
        Path("api/minilm_cls_best.pt"),
        Path("minilm_cls_best.pt"),
    ]
    ckpt = next((p for p in ckpt_candidates if p.exists()), None)
    assert ckpt is not None, "missing checkpoint for evaluation"

    model = MiniLMClassifier().to("cpu")
    model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
    model.eval()

    # Load a very small sample to keep CI fast
    y_true, texts = [], []
    with open("data/valid.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 20:
                break
            obj = json.loads(line)
            y_true.append(int(obj["label"]))
            texts.append(obj["text"])

    preds, _ = predict_batch(model, "cpu", texts, max_len=256)
    cm = confusion_matrix(y_true, preds, num_labels=3)
    # sanity: confusion matrix has correct shape and counts
    assert len(cm) == 3 and all(len(r) == 3 for r in cm)
    assert sum(sum(r) for r in cm) == len(texts)

