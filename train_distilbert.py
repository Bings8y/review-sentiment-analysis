"""Fine-tune DistilBERT (distilbert-base-uncased) for 3-class review sentiment.

Kept deliberately light for a laptop: 128-token reviews, batch 32, 2 epochs, Apple MPS / CUDA / CPU.
The checkpoint with the best validation macro-F1 is kept and evaluated once on the test split.

    python train_distilbert.py        # writes models/distilbert-sentiment/, results/distilbert.json
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from sentiment.metrics import LABELS, evaluate
from sentiment.preprocess import light_clean

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "distilbert-base-uncased"
OUT_DIR = os.path.join(ROOT, "models", "distilbert-sentiment")


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def batches(tok, texts, labels, bs, max_len, shuffle, seed=0):
    idx = np.arange(len(texts))
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        enc = tok([texts[k] for k in j], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([labels[k] for k in j])
        yield enc


@torch.no_grad()
def predict_proba(model, tok, texts, device, bs=64, max_len=128):
    model.eval()
    out = []
    for enc in batches(tok, texts, [0] * len(texts), bs, max_len, shuffle=False):
        enc.pop("labels")
        logits = model(**{k: v.to(device) for k, v in enc.items()}).logits
        out.append(torch.softmax(logits.float(), -1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--max-train", type=int, default=0, help="subsample the training split (0 = all)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    df = pd.read_csv(os.path.join(ROOT, "data", "reviews.csv"))
    df["text"] = df["review"].map(light_clean)
    lab = {c: i for i, c in enumerate(LABELS)}
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    if args.max_train:
        tr = tr.sample(n=min(args.max_train, len(tr)), random_state=args.seed)
    xtr, ytr = tr["text"].tolist(), tr["label"].map(lab).tolist()

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, id2label=dict(enumerate(LABELS)), label2id=lab).to(device)
    print(f"device={device}  train={len(xtr):,}  params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    steps = args.epochs * ((len(xtr) + args.bs - 1) // args.bs)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)

    log, best_f1, step, t0 = {"step": [], "loss": [], "epoch_val": []}, -1, 0, time.time()
    for ep in range(args.epochs):
        model.train()
        for enc in batches(tok, xtr, ytr, args.bs, args.max_len, shuffle=True, seed=args.seed + ep):
            loss = model(**{k: v.to(device) for k, v in enc.items()}).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 10 == 0:
                log["step"].append(step)
                log["loss"].append(round(loss.item(), 4))
            if step % 100 == 0:
                print(f"  epoch {ep + 1} step {step}/{steps} loss {loss.item():.4f}  ({time.time() - t0:.0f}s)")
        pv = predict_proba(model, tok, va["text"].tolist(), device, max_len=args.max_len)
        m = evaluate(va["label"], [LABELS[i] for i in pv.argmax(1)])
        log["epoch_val"].append(dict(epoch=ep + 1, step=step, accuracy=m["accuracy"], macro_f1=m["macro_f1"]))
        print(f"epoch {ep + 1}: val acc={m['accuracy']:.4f} macro-F1={m['macro_f1']:.4f}")
        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            model.save_pretrained(OUT_DIR)
            tok.save_pretrained(OUT_DIR)
    train_s = time.time() - t0

    model = AutoModelForSequenceClassification.from_pretrained(OUT_DIR).to(device)
    t = time.time()
    pt = predict_proba(model, tok, te["text"].tolist(), device, max_len=args.max_len)
    infer_ms = (time.time() - t) / len(te) * 1000
    yp = [LABELS[i] for i in pt.argmax(1)]
    m = evaluate(te["label"], yp)
    m.update(train_seconds=round(train_s, 1), infer_ms_per_review=round(infer_ms, 3), val_macro_f1=best_f1,
             device=str(device), config=vars(args), train_size=len(xtr))
    print(f"TEST DistilBERT acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f}  (trained in {train_s / 60:.1f} min)")

    json.dump(dict(metrics=m, log=log), open(os.path.join(ROOT, "results", "distilbert.json"), "w"), indent=1)
    preds_path = os.path.join(ROOT, "results", "test_predictions.csv")
    preds = pd.read_csv(preds_path)
    preds["DistilBERT"] = yp
    preds["DistilBERT_conf"] = pt.max(1).round(4)
    preds.to_csv(preds_path, index=False)


if __name__ == "__main__":
    main()
