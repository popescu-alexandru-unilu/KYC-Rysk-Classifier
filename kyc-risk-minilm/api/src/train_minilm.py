# src/train_minilm.py
import json, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from model_minilm import MiniLMClassifier, NUM_LABELS  # <<< CHANGE in model_minilm.py if needed
from tqdm.auto import tqdm  # >>> ADDED: tqdm progress bars

BATCH_SIZE = 16   # <<< CHANGE (try 8 or 4 if out of memory)
EPOCHS = 3        # <<< CHANGE as you like

class DS(Dataset):
    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            self.rows = [json.loads(l) for l in f]
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        return r["text"], int(r["label"])

def collate(batch):
    texts  = [t for t,_ in batch]
    labels = torch.tensor([y for _,y in batch], dtype=torch.long)
    return texts, labels

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MiniLMClassifier().to(device)

    # head-only warmup
    for p in model.enc.parameters(): p.requires_grad = False

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4, weight_decay=0.01)
    train = DataLoader(DS("data/train.jsonl"), batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate)
    valid = DataLoader(DS("data/valid.jsonl"), batch_size=BATCH_SIZE*2, shuffle=False, collate_fn=collate)

    train_rows = [json.loads(l) for l in open("data/train.jsonl", encoding="utf-8")]
    counts = Counter(int(r["label"]) for r in train_rows)
    total = sum(counts.values())
    weights = torch.tensor(
        [total / max(1, counts.get(i, 1)) for i in range(NUM_LABELS)],
        dtype=torch.float, device=device
    )
    weights = weights / weights.sum() * NUM_LABELS  # normalize around 1.0

    best = 0.0
    for epoch in range(1, EPOCHS+1):
        # ---- train ----
        model.train(); tot=cor=0; loss_sum=0.0
        pbar = tqdm(train, desc=f"Epoch {epoch} [train]", total=len(train), leave=False)
        for texts, y in pbar:
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                logits = model(texts, device=device)
                loss   = F.cross_entropy(logits, y.to(device), weight=weights, label_smoothing=0.05)  # <<< CHANGE
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            # metrics
            bs = y.size(0)
            tot += bs
            cor += (logits.argmax(-1).detach().cpu()==y).sum().item()
            loss_sum += loss.item()*bs
            pbar.set_postfix(loss=f"{loss_sum/tot:.4f}", acc=f"{cor/tot:.3f}")

        print(f"epoch {epoch} | train loss {loss_sum/tot:.4f} acc {cor/tot:.3f}")

        # ---- valid ----
        model.eval(); tot=cor=0
        pbar_v = tqdm(valid, desc=f"Epoch {epoch} [valid]", total=len(valid), leave=False)
        with torch.no_grad():
            for texts, y in pbar_v:
                logits = model(texts, device=device)
                bs = y.size(0)
                cor += (logits.argmax(-1).detach().cpu()==y).sum().item()
                tot += bs
                pbar_v.set_postfix(acc=f"{cor/tot:.3f}")

        acc = cor/max(1,tot)
        print(f"epoch {epoch} | valid acc {acc:.3f}")
        if acc > best:
            best = acc
            torch.save(model.state_dict(), "minilm_cls_best.pt")
            print("saved minilm_cls_best.pt")

    # OPTIONAL: unfreeze encoder for 1–2 extra epochs (tiny finetune)
    print("Unfreezing encoder for fine-tuning...")
    for p in model.enc.parameters(): p.requires_grad = True
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    for epoch in range(EPOCHS + 1, EPOCHS + 3):
        # ---- train ----
        model.train(); tot=cor=0; loss_sum=0.0
        pbar = tqdm(train, desc=f"Epoch {epoch} [train unfrozen]", total=len(train), leave=False)
        for texts, y in pbar:
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                logits = model(texts, device=device)
                loss   = F.cross_entropy(logits, y.to(device), weight=weights, label_smoothing=0.05)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            bs = y.size(0)
            tot += bs
            cor += (logits.argmax(-1).detach().cpu()==y).sum().item()
            loss_sum += loss.item()*bs
            pbar.set_postfix(loss=f"{loss_sum/tot:.4f}", acc=f"{cor/tot:.3f}")
        print(f"epoch {epoch} | train loss {loss_sum/tot:.4f} acc {cor/tot:.3f}")

        # ---- valid ----
        model.eval(); tot=cor=0
        pbar_v = tqdm(valid, desc=f"Epoch {epoch} [valid unfrozen]", total=len(valid), leave=False)
        with torch.no_grad():
            for texts, y in pbar_v:
                logits = model(texts, device=device)
                bs = y.size(0)
                cor += (logits.argmax(-1).detach().cpu()==y).sum().item()
                tot += bs
                pbar_v.set_postfix(acc=f"{cor/tot:.3f}")
        acc = cor/max(1,tot)
        print(f"epoch {epoch} | valid acc {acc:.3f}")
        if acc > best:
            best = acc
            torch.save(model.state_dict(), "minilm_cls_best.pt")
            print("saved minilm_cls_best.pt")
    print(f"Final best valid acc: {best:.3f}")

if __name__ == "__main__":
    main()
