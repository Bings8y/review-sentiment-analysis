"""Text preprocessing for the classical (TF-IDF) models.

Pipeline: lowercase -> expand contractions -> strip URLs/HTML/punctuation/digits -> tokenise
-> remove stopwords (negation-aware) -> lemmatise.

The standard NLTK English stopword list contains "not", "no", "nor", "don't", "isn't" and similar.
Removing them turns "not good" into "good", so by default they are kept (see NEGATION_WORDS).
"""
import re
from functools import lru_cache

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

NEGATION_WORDS = {"not", "no", "nor", "never", "none", "nothing", "nobody", "neither", "nowhere",
                  "cannot", "without", "but", "against", "too", "very", "only"}
STANDARD_STOPWORDS = set(stopwords.words("english"))
NEGATION_AWARE_STOPWORDS = (STANDARD_STOPWORDS - NEGATION_WORDS) |{"would", "could", "also", "get", "got", "one", "product", "use", "used"}

CONTRACTIONS = [
    (r"won't", "will not"), (r"can't", "can not"), (r"shan't", "shall not"), (r"ain't", "is not"),
    (r"n't", " not"), (r"'re", " are"), (r"'s", " is"), (r"'d", " would"), (r"'ll", " will"),
    (r"'ve", " have"), (r"'m", " am"),
]
URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<[^>]+>")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")

_lemmatizer = WordNetLemmatizer()


@lru_cache(maxsize=200_000)
def lemmatize(word):
    # verb first ("loved" -> "love", "was" -> "be"), then noun ("bottles" -> "bottle")
    return _lemmatizer.lemmatize(_lemmatizer.lemmatize(word, "v"), "n")


def clean_text(text, stop=NEGATION_AWARE_STOPWORDS, lemma=True):
    text = str(text).lower().replace("’", "'")
    for pat, rep in CONTRACTIONS:
        text = re.sub(pat, rep, text)
    text = URL_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    tokens = [t for t in word_tokenize(text) if len(t) > 1 or t == "i"]
    tokens = [t for t in tokens if t not in stop]
    if lemma:
        tokens = [lemmatize(t) for t in tokens]
    return " ".join(tokens)


def clean_standard(text):
    """Variant with the unmodified NLTK stopword list (used only for the negation ablation)."""
    return clean_text(text, stop=STANDARD_STOPWORDS)


def light_clean(text):
    """Minimal cleaning for DistilBERT, which needs the original word order and function words."""
    text = HTML_RE.sub(" ", str(text))
    text = URL_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()
