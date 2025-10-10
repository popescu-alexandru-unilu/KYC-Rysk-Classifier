# src/train_minilm.py
import os, time, json, argparse, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from model_minilm import MiniLMClassifier, NUM_LABELS  # <<< CHANGE in model_minilm.py if needed
from tqdm.auto import tqdm  # >>> ADDED: tqdm progress bars

BATCH_SIZE = 16   # default; can be overridden via args
EPOCHS = 3        # default; can be overridden via args

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

def focal_loss(logits, targets, alpha=None, gamma: float = 2.0):
    """Focal loss on logits. alpha: tensor [C] or float.
    logits: [B, C], targets: [B].
    """
    ce = F.cross_entropy(logits, targets, reduction='none')  # [B]
    pt = torch.exp(-ce)
    if alpha is not None:
        if isinstance(alpha, torch.Tensor):
            a = alpha.to(logits.device)[targets]
        else:
            a = torch.full_like(pt, float(alpha))
        loss = a * ((1 - pt) ** gamma) * ce
    else:
        loss = ((1 - pt) ** gamma) * ce
    return loss.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--loss', choices=['ce','focal'], default='ce')
    ap.add_argument('--focal_gamma', type=float, default=2.0)
    ap.add_argument('--class_weights', type=str, default='auto', help='comma list or "auto"')
    ap.add_argument('--train', default='data/train.jsonl')
    ap.add_argument('--valid', default='data/valid.jsonl')
    ap.add_argument('--metrics_dir', default='', help='If set, write JSON metrics per epoch into this directory')
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MiniLMClassifier().to(device)

    # head-only warmup
    for p in model.enc.parameters(): p.requires_grad = False

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4, weight_decay=0.01)
    train = DataLoader(DS(args.train), batch_size=args.batch_size, shuffle=True,  collate_fn=collate)
    valid = DataLoader(DS(args.valid), batch_size=args.batch_size*2, shuffle=False, collate_fn=collate)

    # class weights
    if args.class_weights == 'auto':
        train_rows = [json.loads(l) for l in open(args.train, encoding="utf-8")]
        counts = Counter(int(r["label"]) for r in train_rows)
        total = sum(counts.values())
        weights = torch.tensor(
            [total / max(1, counts.get(i, 1)) for i in range(NUM_LABELS)],
            dtype=torch.float, device=device
        )
        weights = weights / weights.sum() * NUM_LABELS
    else:
        vals = [float(x) for x in args.class_weights.split(',')]
        assert len(vals) == NUM_LABELS, f"class_weights must have {NUM_LABELS} values"
        weights = torch.tensor(vals, dtype=torch.float, device=device)

    best = 0.0
    for epoch in range(1, args.epochs+1):
        # ---- train ----
        model.train(); tot=cor=0; loss_sum=0.0
        pbar = tqdm(train, desc=f"Epoch {epoch} [train]", total=len(train), leave=False)
        for texts, y in pbar:
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                logits = model(texts, device=device)
                if args.loss == 'focal':
                    loss = focal_loss(logits, y.to(device), alpha=weights, gamma=args.focal_gamma)
                else:
                    loss = F.cross_entropy(logits, y.to(device), weight=weights, label_smoothing=0.05)
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
        cm = [[0]*NUM_LABELS for _ in range(NUM_LABELS)]
        pbar_v = tqdm(valid, desc=f"Epoch {epoch} [valid]", total=len(valid), leave=False)
        with torch.no_grad():
            for texts, y in pbar_v:
                logits = model(texts, device=device)
                bs = y.size(0)
                preds = logits.argmax(-1).detach().cpu()
                cor += (preds==y).sum().item()
                for yt, yp in zip(y.tolist(), preds.tolist()):
                    cm[yt][yp] += 1
                tot += bs
                pbar_v.set_postfix(acc=f"{cor/tot:.3f}")

        acc = cor/max(1,tot)
        # per-class precision/recall and macro-F1
        prec = []
        rec  = []
        f1   = []
        for i in range(NUM_LABELS):
            tp = cm[i][i]
            fp = sum(cm[r][i] for r in range(NUM_LABELS) if r != i)
            fn = sum(cm[i][c] for c in range(NUM_LABELS) if c != i)
            p = tp / max(1, (tp + fp))
            r = tp / max(1, (tp + fn))
            prec.append(p); rec.append(r)
            f1.append(0.0 if p+r==0 else 2*p*r/(p+r))
        macro_f1 = sum(f1)/NUM_LABELS
        print(f"epoch {epoch} | valid acc {acc:.3f} macroF1 {macro_f1:.3f}")
        print("per-class (idx):", {i: {"prec": round(prec[i],3), "rec": round(rec[i],3), "f1": round(f1[i],3)} for i in range(NUM_LABELS)})
        if acc > best:
            best = acc
            torch.save(model.state_dict(), "minilm_cls_best.pt")
            print("saved minilm_cls_best.pt")

        # write metrics artifact
        if args.metrics_dir:
            os.makedirs(args.metrics_dir, exist_ok=True)
            artifact = {
                "ts": time.time(),
                "epoch": epoch,
                "accuracy": acc,
                "macro_f1": macro_f1,
                "per_class": {i: {"prec": prec[i], "rec": rec[i], "f1": f1[i]} for i in range(NUM_LABELS)},
                "confusion": cm,
            }
            path = os.path.join(args.metrics_dir, f"valid_metrics_epoch_{epoch}.json")
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(artifact, f)
            print(f"wrote {path}")

    # OPTIONAL: unfreeze encoder for 1–2 extra epochs (tiny finetune)
    print("Unfreezing encoder for fine-tuning...")
    for p in model.enc.parameters(): p.requires_grad = True
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    for epoch in range(args.epochs + 1, args.epochs + 3):
        # ---- train ----
        model.train(); tot=cor=0; loss_sum=0.0
        pbar = tqdm(train, desc=f"Epoch {epoch} [train unfrozen]", total=len(train), leave=False)
        for texts, y in pbar:
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                logits = model(texts, device=device)
                if args.loss == 'focal':
                    loss = focal_loss(logits, y.to(device), alpha=weights, gamma=args.focal_gamma)
                else:
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
