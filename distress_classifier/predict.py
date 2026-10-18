"""
Single-text inference: predicts which category best matches the text.

Returns the predicted category (e.g. Depression, Anxiety) and a probability
for each of the 5 categories. Works in Cebuano, Tagalog, and English.

ETHICAL NOTE: This classifies LINGUISTIC PATTERNS only. It does NOT diagnose
mental illness. Research and screening support only.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
from tensorflow import keras

from categories import CATEGORIES
from preprocess import MAX_SEQUENCE_LENGTH, clean_text, texts_to_sequences


UNCERTAIN_LABEL = "Uncertain"
DEFAULT_UNCERTAIN_THRESHOLD = 0.70


def load_model_and_tokenizer(save_dir: Union[str, Path]):
    """Load trained model, tokenizer, and category list from output directory."""
    save_dir = Path(save_dir)
    model_path = save_dir / "best_model.keras"
    tokenizer_path = save_dir / "tokenizer.pkl"
    config_path = save_dir / "tokenizer_config.json"
    categories_path = save_dir / "categories.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    model = keras.models.load_model(model_path)
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    maxlen = MAX_SEQUENCE_LENGTH
    if config_path.exists():
        with open(config_path) as f:
            maxlen = json.load(f).get("max_sequence_length", MAX_SEQUENCE_LENGTH)

    categories = CATEGORIES
    if categories_path.exists():
        with open(categories_path) as f:
            categories = json.load(f)

    return model, tokenizer, maxlen, categories


def predict(
    text: str,
    model=None,
    tokenizer=None,
    categories: List[str] = None,
    maxlen: int = MAX_SEQUENCE_LENGTH,
    model_dir: Union[str, Path, None] = "output",
    uncertain_threshold: float = DEFAULT_UNCERTAIN_THRESHOLD,
) -> Dict:
    """
    Predict the category and per-class probabilities for the input text.

    Args:
        text: Raw input string (any of Cebuano, Tagalog, English).
        model, tokenizer, categories, maxlen: Optional; if omitted, loaded from model_dir.
        model_dir: Directory with best_model.keras, tokenizer.pkl, categories.json.
        uncertain_threshold: If top probability is below this value, return
            "Uncertain" as predicted_category. Set to <=0 to disable.

    Returns:
        Dict with:
          - "predicted_category": str (e.g. "Depression" or "Uncertain")
          - "probability": float (confidence for top class, 0-1)
          - "all_probabilities": { "Depression": 0.5, "Anxiety": 0.3, ... }
          - "is_uncertain": bool
          - "uncertain_threshold": float
          - "top_class": str (raw top class before uncertainty rule)
    """
    if model is None or tokenizer is None or categories is None:
        model, tokenizer, maxlen, categories = load_model_and_tokenizer(model_dir)

    if uncertain_threshold is None:
        uncertain_threshold = DEFAULT_UNCERTAIN_THRESHOLD

    cleaned = clean_text(text)
    if not cleaned:
        base_p = 1.0 / len(categories)
        top_class = categories[0]
        is_uncertain = uncertain_threshold > 0 and base_p < uncertain_threshold
        return {
            "predicted_category": UNCERTAIN_LABEL if is_uncertain else top_class,
            "probability": base_p,
            "all_probabilities": {c: base_p for c in categories},
            "is_uncertain": is_uncertain,
            "uncertain_threshold": float(uncertain_threshold),
            "top_class": top_class,
        }

    seq = texts_to_sequences(tokenizer, [cleaned], maxlen=maxlen)
    proba = model.predict(seq, verbose=0)[0]
    idx = int(np.argmax(proba))
    top_prob = float(proba[idx])
    top_class = categories[idx]

    num_classes = len(proba)
    all_probs = {categories[i]: float(proba[i]) for i in range(min(num_classes, len(categories)))}

    is_uncertain = uncertain_threshold > 0 and top_prob < uncertain_threshold
    predicted = UNCERTAIN_LABEL if is_uncertain else top_class

    return {
        "predicted_category": predicted,
        "probability": top_prob,
        "all_probabilities": all_probs,
        "is_uncertain": is_uncertain,
        "uncertain_threshold": float(uncertain_threshold),
        "top_class": top_class,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=str, nargs="?", default="I have been feeling really down and alone lately.")
    parser.add_argument("--model_dir", type=str, default="output")
    parser.add_argument(
        "--uncertain_threshold",
        type=float,
        default=DEFAULT_UNCERTAIN_THRESHOLD,
        help="Return 'Uncertain' when top probability is below this value. Set <=0 to disable.",
    )
    args = parser.parse_args()

    result = predict(
        args.text,
        model_dir=args.model_dir,
        uncertain_threshold=args.uncertain_threshold,
    )

    print(f"Input: {args.text[:80]}{'...' if len(args.text) > 80 else ''}")
    print(f"Predicted category: {result['predicted_category']} (probability: {result['probability']:.4f})")
    if result["is_uncertain"]:
        print(
            f"Top class was '{result['top_class']}' but below threshold "
            f"{result['uncertain_threshold']:.2f}. Marked as {UNCERTAIN_LABEL}."
        )
    print("All categories:")
    for cat, p in sorted(result["all_probabilities"].items(), key=lambda x: -x[1]):
        print(f"  {cat}: {p:.4f}")
    print("(This is a research indicator only; not a mental health diagnosis.)")
