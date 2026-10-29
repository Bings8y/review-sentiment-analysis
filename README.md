# Sentiment Analysis of Customer Product Reviews Using NLP

Classifies Amazon product reviews as **positive, neutral or negative**. TF-IDF features with Multinomial
Naive Bayes, Logistic Regression and a linear SVM are compared with a fine-tuned DistilBERT, and the best
models are served by a Streamlit app that returns the label with a confidence score.

## How it works

```
Amazon reviews ─► clean + label ─► preprocess ─► TF-IDF (1–2 grams) ─► NB / LR / SVM ─┐
 (30k, star → label) (80/10/10 split) (negation-aware)                                     ├─► evaluate ─► Streamlit app
                                    └──────── light clean ─► DistilBERT fine-tune ──────────┘
```

1. **Data** (`prepare_data.py`): downloads ~40 MB of Amazon Reviews 2023 (All_Beauty) from Hugging Face.
   Labels come from the star rating: 1–2 ★ negative, 3 ★ neutral, 4–5 ★ positive. Auto-filled titles such as
   "Five Stars" are removed because they leak the label. Stratified 80/10/10 train/validation/test split.
2. **Preprocessing** (`sentiment/preprocess.py`): lowercase, expand contractions, strip URLs/HTML/punctuation,
   NLTK tokenise, remove stopwords **but keep negations** (not, no, never, but, …), WordNet lemmatise.
3. **Classical models** (`train_classical.py`): TF-IDF with unigrams + bigrams (~60k features). Naive Bayes,
   Logistic Regression and linear SVM are tuned on the validation set by macro-F1; LR and SVM are also trained
   with `class_weight="balanced"` to handle the small neutral class. Includes a negation ablation.
4. **DistilBERT** (`train_distilbert.py`): fine-tunes `distilbert-base-uncased` (128 tokens, batch 32, 2 epochs,
   AdamW 3e-5, linear warm-up). Uses CUDA, Apple MPS or CPU automatically.
5. **Evaluation** (`sentiment/metrics.py`, `make_figures.py`): accuracy, macro / per-class precision-recall-F1,
   confusion matrices, error analysis; figures go to `results/figures/`.
6. **Demo** (`app.py`, `sentiment/predict.py`): type a review, pick a model, get the label and confidence.

## Layout

| Path | Purpose |
|---|---|
| `prepare_data.py` | Download, clean, label and split the reviews → `data/reviews.csv` |
| `sentiment/preprocess.py` | Text preprocessing for the TF-IDF models |
| `sentiment/metrics.py` | Shared evaluation metrics |
| `train_classical.py` | TF-IDF + NB / LR / SVM, grid search, ablation → `models/*.joblib`, `results/classical.json` |
| `train_distilbert.py` | DistilBERT fine-tuning → `models/distilbert-sentiment/`, `results/distilbert.json` |
| `make_figures.py` | Figures and error analysis → `results/figures/`, `results/analysis.json` |
| `sentiment/predict.py`, `app.py` | Inference and the Streamlit demo |
| `run_all.bat` | Whole pipeline on Windows |

`data/reviews.csv` and the TF-IDF models are included, so the app works straight after installing.
The DistilBERT model (256 MB) is too large for GitHub; train it with step 5 below to enable it in the app.

## Run on Windows

Requires **Python 3.10–3.13** from [python.org](https://www.python.org/downloads/) (tick *"Add python.exe to PATH"*).
Open **Command Prompt** in the project folder:

```bat
:: 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

:: 2. Install packages
pip install -r requirements.txt
python -c "import nltk; [nltk.download(p) for p in ('stopwords','wordnet','omw-1.4','punkt_tab')]"

:: 3. Run the app straight away (TF-IDF models are included) - opens http://localhost:8501
streamlit run app.py
```

To rebuild everything from scratch:

```bat
python prepare_data.py                          :: download + split the data
python train_classical.py                       :: NB / LR / SVM
python train_distilbert.py --max-train 12000    :: DistilBERT (slow on CPU, see below)
python make_figures.py                          :: figures + error analysis
streamlit run app.py
```

or simply `run_all.bat`.

**Windows notes**

- In **PowerShell** activate with `.venv\Scripts\Activate.ps1`. If it says scripts are disabled, run
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use Command Prompt instead.
- **No NVIDIA GPU:** DistilBERT trains on the CPU and can take an hour or more. For a quicker run use
  `python train_distilbert.py --max-train 3000 --epochs 1` (lower accuracy), or skip it; the app still works
  with the TF-IDF models.
- **NVIDIA GPU:** install the CUDA build of PyTorch first, then the rest:
  `pip install torch --index-url https://download.pytorch.org/whl/cu121` then `pip install -r requirements.txt`.
  The script picks the GPU automatically.
- If `python` opens the Microsoft Store, use `py` instead (e.g. `py -m venv .venv`).

## Run on macOS / Linux

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; [nltk.download(p) for p in ('stopwords','wordnet','omw-1.4','punkt_tab')]"
streamlit run app.py
```

## Results (test split, 3,000 reviews)

| Model | Accuracy | Macro F1 | Neutral recall |
|---|---|---|---|
| Multinomial Naive Bayes | 0.844 | 0.593 | 0.063 |
| Logistic Regression | 0.869 | 0.667 | 0.170 |
| Linear SVM | 0.868 | 0.662 | 0.158 |
| Logistic Regression (class-weighted) | 0.839 | 0.703 | **0.478** |
| Linear SVM (class-weighted) | 0.860 | 0.683 | 0.261 |
| **DistilBERT** (12k reviews, 2 epochs, 5.3 min on Apple M4 Pro) | **0.882** | **0.722** | 0.344 |

* The data is imbalanced (70% positive, 22% negative, 8% neutral), so **macro F1** is the main metric.
* Keeping negation words in preprocessing adds 0.053 macro F1 (0.614 → 0.667, LR with uni+bigrams).
* 10.8% of raw reviews have auto-generated titles ("Five Stars") that leak the label; they are removed.
* Remaining errors are mostly neutral, mixed and praise-word negative ("I loved it until it died") reviews.
