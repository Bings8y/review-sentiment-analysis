"""Download a sample of Amazon product reviews and build the labelled train/val/test splits.

Source: Amazon Reviews 2023 (Hou et al., 2024), category "All_Beauty", hosted on Hugging Face.
The full file is 327 MB, so only a few byte ranges spread evenly across it are fetched (about 40 MB);
the reviews in the file are grouped by user, so spreading the ranges gives a varied sample.

Star rating -> sentiment: 1-2 negative, 3 neutral, 4-5 positive.

    python prepare_data.py            # writes data/reviews.csv (with a 'split' column)
"""
import argparse
import json
import os
import urllib.request

import pandas as pd
from sklearn.model_selection import train_test_split

URL = ("https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/"
       "raw/review_categories/All_Beauty.jsonl")
FILE_SIZE = 326_611_506
ROOT = os.path.dirname(os.path.abspath(__file__))
LABELS = ["negative", "neutral", "positive"]


def rating_to_label(r):
    return "negative" if r <= 2 else ("neutral" if r == 3 else "positive")


def fetch_range(start, length):
    req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{start + length - 1}"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
    lines = raw.split("\n")[1:-1]  # first and last lines are cut by the byte range
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--chunk-mb", type=float, default=5.0)
    ap.add_argument("--n", type=int, default=30000, help="reviews kept after cleaning")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    cache = os.path.join(ROOT, "data", "raw_sample.jsonl")
    if not os.path.exists(cache):
        rows, size = [], int(args.chunk_mb * 1e6)
        step = (FILE_SIZE - size) // (args.chunks - 1)
        for k in range(args.chunks):
            part = fetch_range(k * step, size)
            rows += part
            print(f"chunk {k + 1}/{args.chunks}: {len(part):,} reviews")
        with open(cache, "w") as f:
            for r in rows:
                f.write(json.dumps({key: r.get(key) for key in ("rating", "title", "text", "parent_asin",
                                                                 "verified_purchase", "helpful_vote")}) + "\n")
    df = pd.read_json(cache, lines=True)
    print(f"raw reviews: {len(df):,}")

    df["title"] = df["title"].fillna("").str.strip()
    # Amazon auto-fills empty titles with "Five Stars", "Two Stars", ...: that is the label itself, so drop it
    df.loc[df["title"].str.fullmatch(r"(?i)(one|two|three|four|five) stars?"), "title"] = ""
    df["text"] = df["text"].fillna("").str.replace(r"<br\s*/?>", " ", regex=True).str.strip()
    df["review"] = (df["title"] + ". " + df["text"]).str.strip(". ")
    df = df[df["text"].str.split().str.len() >= 3]
    df = df.drop_duplicates(subset="review")
    df["label"] = df["rating"].astype(int).map(rating_to_label)
    print(f"after cleaning: {len(df):,}")

    df = df.sample(n=min(args.n, len(df)), random_state=args.seed).reset_index(drop=True)
    train, rest = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=args.seed)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["label"], random_state=args.seed)
    df["split"] = "train"
    df.loc[val.index, "split"] = "val"
    df.loc[test.index, "split"] = "test"
    out = df[["review", "rating", "label", "split", "parent_asin", "verified_purchase"]]
    out.to_csv(os.path.join(ROOT, "data", "reviews.csv"), index=False)

    print(out.groupby(["split", "label"]).size().unstack())
    print("rating distribution:", out["rating"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
