"""
Text preprocessing for distress-pattern classification.

Designed for multilingual input: Cebuano, Tagalog, and English. Cleaning is
language-agnostic (lowercase, whitespace normalization) and preserves Unicode
letters (e.g. Tagalog/Cebuano characters) for tokenization.

ETHICAL NOTE: This module supports research on linguistic patterns associated with
self-reported distress in text. It does NOT constitute clinical assessment or
diagnosis. Outputs are research indicators only, not medical advice.
"""

import re
from typing import List

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ---------------------------------------------------------------------------
# Preprocessing constants (tune for your corpus; these are thesis defaults)
# ---------------------------------------------------------------------------
MAX_SEQUENCE_LENGTH = 128   # Max tokens per document after padding/truncation
MAX_VOCAB_SIZE = 20000      # Vocabulary size (OOV mapped to reserved index)
OOV_TOKEN = "<OOV>"
PADDING = "post"
TRUNCATING = "post"

# Lightweight lexical normalization for common variants/slang forms.
NORMALIZATION_RULES = [
    (r"\bdepress(?:ed|ing)?\b", "depression"),
    (r"\bdepres\b", "depression"),
    (r"\banxious\b", "anxiety"),
    (r"\bstressed\b", "stress"),
]


def clean_text(text: str) -> str:
    """
    Light cleaning: lowercasing and normalization only.
    Safe for Cebuano, Tagalog, and English (preserves Unicode letters, no
    language-specific stemming). Avoids over-sanitizing.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Lowercase for consistent matching with vocabulary (works for en/tl/ceb)
    text = text.lower().strip()

    # Normalize common lexical variants to strengthen class signal.
    for pattern, replacement in NORMALIZATION_RULES:
        text = re.sub(pattern, replacement, text)

    # Collapse repeated whitespace
    text = re.sub(r"\s+", " ", text)
    return text


def fit_tokenizer(texts: List[str]) -> Tokenizer:
    """
    Build vocabulary from training texts only (no data leakage).
    Tokenizer is language-agnostic; use with Cebuano, Tagalog, and English.
    """
    tokenizer = Tokenizer(
        num_words=MAX_VOCAB_SIZE,
        oov_token=OOV_TOKEN,
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        lower=True,
        split=" ",
    )
    tokenizer.fit_on_texts(texts)
    return tokenizer


def texts_to_sequences(
    tokenizer: Tokenizer,
    texts: List[str],
    maxlen: int = MAX_SEQUENCE_LENGTH,
) -> np.ndarray:
    """Convert list of strings to padded integer sequences."""
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(
        sequences,
        maxlen=maxlen,
        padding=PADDING,
        truncating=TRUNCATING,
        value=0,  # padding index (same as tokenizer's 0)
    )


def get_preprocessor_state(tokenizer: Tokenizer) -> dict:
    """Capture tokenizer config for reuse at inference (e.g., in predict.py)."""
    return {
        "word_index": tokenizer.word_index,
        "num_words": getattr(tokenizer, "num_words", MAX_VOCAB_SIZE),
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
    }
