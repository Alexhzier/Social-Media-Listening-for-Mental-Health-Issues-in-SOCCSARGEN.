"""
Training and evaluation for multi-class distress-pattern classifier.

Classifies text into one of 5 categories: Depression, Stress, Anxiety, Addiction,
Suicidal Behavior. Supports Cebuano, Tagalog, and English.

ETHICAL NOTE: This pipeline detects LINGUISTIC PATTERNS only. It does NOT perform
mental health diagnosis. Research indicators only.
"""

import argparse
import json
import pickle
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.error import HTTPError

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from categories import CATEGORIES, category_to_index
from data import SAMPLES
from data_loader import build_google_sheet_url, load_from_google_sheet
from model import NUM_CLASSES, build_cnn_lstm
from preprocess import (
    MAX_SEQUENCE_LENGTH,
    MAX_VOCAB_SIZE,
    clean_text,
    fit_tokenizer,
    texts_to_sequences,
)


def _save_training_curves(history, save_dir: Path) -> None:
    """
    Save training history and curve images.
    - Always saves training_history.json
    - Saves loss_curve.png / accuracy_curve.png if matplotlib is available
    """
    hist = dict(getattr(history, "history", {}) or {})
    with open(save_dir / "training_history.json", "w") as f:
        json.dump(hist, f, indent=2)

    if not hist:
        print("Warning: No training history captured; skipping curve plots.")
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Warning: Could not generate plot images (matplotlib unavailable): {e}")
        return

    # Loss curve
    loss = hist.get("loss", [])
    val_loss = hist.get("val_loss", [])
    if loss:
        epochs = range(1, len(loss) + 1)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, loss, label="Train Loss", linewidth=2)
        if val_loss:
            ax.plot(epochs, val_loss, label="Validation Loss", linewidth=2)
        ax.set_title("Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(save_dir / "loss_curve.png", dpi=150)
        plt.close(fig)

    # Accuracy curve
    acc = hist.get("accuracy", [])
    val_acc = hist.get("val_accuracy", [])
    if acc:
        epochs = range(1, len(acc) + 1)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, acc, label="Train Accuracy", linewidth=2)
        if val_acc:
            ax.plot(epochs, val_acc, label="Validation Accuracy", linewidth=2)
        ax.set_title("Training Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(save_dir / "accuracy_curve.png", dpi=150)
        plt.close(fig)


def _sheet_arg_to_csv_url(sheet_arg: str, fallback_gid: int = 0) -> str:
    """
    Convert sheet ID, /edit URL, or CSV URL into Google CSV export URL.
    """
    value = sheet_arg.strip()
    if not value.startswith("http"):
        return build_google_sheet_url(value, gid=fallback_gid)

    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", value)
    if not match:
        return value

    sheet_id = match.group(1)
    parsed = urlparse(value)
    query = parse_qs(parsed.query)

    gid = fallback_gid
    if "gid" in query and query["gid"]:
        try:
            gid = int(query["gid"][0])
        except ValueError:
            pass
    elif parsed.fragment.startswith("gid="):
        try:
            gid = int(parsed.fragment.split("=", 1)[1])
        except ValueError:
            pass

    return build_google_sheet_url(sheet_id, gid=gid)


def main():
    parser = argparse.ArgumentParser(description="Train multi-class distress-pattern classifier")
    parser.add_argument("--val_split", type=float, default=0.2, help="Fraction for validation")
    parser.add_argument("--epochs", type=int, default=30, help="Max training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--lstm_units", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="output", help="Directory for model and artifacts")
    parser.add_argument(
        "--google_sheet",
        type=str,
        nargs="*",
        default=None,
        help="Load data from one or more Google Sheets: sheet ID(s) or full CSV URL(s). Can pass multiple: --google_sheet ID1 ID2.",
    )
    parser.add_argument("--sheet_gid", type=int, default=0, help="Sheet tab index when using --google_sheet with sheet IDs (default: 0)")
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Load data: from one or more Google Sheet URL(s)/ID(s), or from data.py
    # -------------------------------------------------------------------------
    if args.google_sheet and len(args.google_sheet) > 0:
        all_texts_raw = []
        all_labels_list = []
        for sheet_arg in args.google_sheet:
            sheet_arg = sheet_arg.strip()
            url = _sheet_arg_to_csv_url(sheet_arg, fallback_gid=args.sheet_gid)
            try:
                texts_raw, labels_list = load_from_google_sheet(url, max_rows=None)
                all_texts_raw.extend(texts_raw)
                all_labels_list.extend(labels_list)
                print(f"Loaded {len(texts_raw)} rows from sheet: {sheet_arg[:20]}...")
            except Exception as e:
                if isinstance(e, HTTPError) and e.code in (401, 403):
                    print(f"Warning: Skipping sheet (not shared for public view): {sheet_arg[:30]}...")
                elif isinstance(e, ValueError):
                    print(f"Warning: Skipping sheet (invalid/empty rows): {sheet_arg[:30]}... ({e})")
                else:
                    raise SystemExit(f"Failed to load Google Sheet {sheet_arg!r}: {e}")
        texts = [clean_text(t) for t in all_texts_raw]
        labels = np.array(all_labels_list, dtype=np.int32)
    else:
        texts = [clean_text(t) for t, key in SAMPLES]
        try:
            labels = np.array([category_to_index(key) for _, key in SAMPLES], dtype=np.int32)
        except ValueError as e:
            raise SystemExit(f"Invalid category in data.py: {e}")

    n = len(texts)
    if n == 0:
        raise ValueError("No samples in dataset. Provide --google_sheet SHEET_ID (or add local samples in data.py).")
    print(f"Total loaded samples: {n}")
    class_counts = np.bincount(labels, minlength=len(CATEGORIES))
    for i, category in enumerate(CATEGORIES):
        print(f"  {category}: {int(class_counts[i])}")

    # Stratified split keeps class proportions in train/validation.
    train_texts, val_texts, y_train, y_val = train_test_split(
        texts,
        labels,
        test_size=args.val_split,
        random_state=args.random_state,
        stratify=labels,
    )

    tokenizer = fit_tokenizer(train_texts)
    vocab_size = min(MAX_VOCAB_SIZE, len(tokenizer.word_index) + 1)
    X_train = texts_to_sequences(tokenizer, train_texts, maxlen=MAX_SEQUENCE_LENGTH)
    X_val = texts_to_sequences(tokenizer, val_texts, maxlen=MAX_SEQUENCE_LENGTH)

    # Balanced class weights reduce bias toward majority classes.
    class_indices = np.arange(len(CATEGORIES))
    present_classes = np.unique(y_train)
    weights_present = compute_class_weight(class_weight="balanced", classes=present_classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(present_classes, weights_present)}
    print(f"Class weights (present classes): {class_weight}")

    # -------------------------------------------------------------------------
    # Build and train multi-class model
    # -------------------------------------------------------------------------
    model = build_cnn_lstm(
        vocab_size=vocab_size,
        num_classes=NUM_CLASSES,
        embedding_dim=args.embedding_dim,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        lstm_units=args.lstm_units,
        learning_rate=args.learning_rate,
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=save_dir / "best_model.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        )
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )
    _save_training_curves(history, save_dir)

    # -------------------------------------------------------------------------
    # Evaluation: accuracy, precision, recall, F1, and confusion matrix
    # -------------------------------------------------------------------------
    y_pred_proba = model.predict(X_val, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = y_val

    accuracy = float(accuracy_score(y_true, y_pred))
    precision_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    precision_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    recall_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CATEGORIES)))

    print("\n--- Validation metrics ---")
    print(f"  accuracy: {accuracy:.4f}")
    print(f"  precision_macro: {precision_macro:.4f}")
    print(f"  recall_macro: {recall_macro:.4f}")
    print(f"  f1_macro: {f1_macro:.4f}")
    print(f"  precision_weighted: {precision_weighted:.4f}")
    print(f"  recall_weighted: {recall_weighted:.4f}")
    print(f"  f1_weighted: {f1_weighted:.4f}")
    print("\n--- Per-class report ---")
    print(classification_report(y_true, y_pred, labels=range(len(CATEGORIES)), target_names=CATEGORIES, zero_division=0))
    print("\n--- Confusion matrix (rows=true, cols=pred) ---")
    print("Classes:", CATEGORIES)
    print(cm)

    metrics = {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "confusion_matrix": {
            "labels": CATEGORIES,
            "matrix": cm.tolist(),
        },
    }

    # Save artifacts for inference
    with open(save_dir / "tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    with open(save_dir / "tokenizer_config.json", "w") as f:
        json.dump({"max_sequence_length": MAX_SEQUENCE_LENGTH}, f)
    with open(save_dir / "categories.json", "w") as f:
        json.dump(CATEGORIES, f, indent=2)
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(save_dir / "confusion_matrix.json", "w") as f:
        json.dump({"labels": CATEGORIES, "matrix": cm.tolist()}, f, indent=2)

    # Quick sanity predictions using in-memory model/tokenizer.
    print("\n--- Testing with sample texts ---")
    from predict import predict

    test_samples = [
        ("na depress nako", "Depression"),
        ("stress kaayo ko", "Stress"),
        ("anxiety nako", "Anxiety"),
        ("addiction nako", "Addiction"),
        ("hikog", "Suicidal Behavior"),
    ]
    for text, expected in test_samples:
        result = predict(
            text,
            model=model,
            tokenizer=tokenizer,
            categories=CATEGORIES,
            maxlen=MAX_SEQUENCE_LENGTH,
            model_dir=save_dir,
        )
        print(f"Text: '{text}'")
        print(f"  Expected: {expected}")
        print(f"  Predicted: {result['predicted_category']} (prob: {result['probability']:.4f})")
        print()

    print(f"Model and artifacts saved to: {save_dir.absolute()}")
    return model, tokenizer, metrics


if __name__ == "__main__":
    main()


