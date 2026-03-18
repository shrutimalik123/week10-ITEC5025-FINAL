"""
train.py
========
Week 8 Assignment – Chatbot Model Development and Training
ITEC5025 – Hypotify Clinical Chatbot
Author: Shruti Malik
Date:   2026-02-28

Purpose:
    Load the preprocessed chatbot dataset, split it into training / validation /
    test sets, build the BiLSTM model defined in model.py, train it with
    callbacks, evaluate on the held-out test set, and save all artefacts.

    Outputs:
        chatbot_model_w8.keras   – trained Keras model
        tokenizer_w8.pkl         – fitted Keras Tokenizer
        label_encoder_w8.pkl     – fitted scikit-learn LabelEncoder
        training_history.csv     – epoch-by-epoch loss and accuracy

Data Split (Stratified):
    70% Training   – used for gradient descent updates
    15% Validation – used by EarlyStopping and ReduceLROnPlateau callbacks
    15% Test       – held out; used only for final evaluation

Usage:
    python train.py
"""

import csv
import json
import logging
import os
import pickle
import sys

# ── Force UTF-8 output on Windows ─────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Third-party imports ───────────────────────────────────────────────────────
try:
    import numpy as np
except ImportError:
    log.error("numpy not found. Run: pip install numpy")
    sys.exit(1)

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.callbacks import (
        EarlyStopping,
        ReduceLROnPlateau,
        ModelCheckpoint,
    )
except ImportError:
    log.error("TensorFlow not found. Run: pip install tensorflow")
    sys.exit(1)

try:
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
except ImportError:
    log.error("scikit-learn not found. Run: pip install scikit-learn")
    sys.exit(1)

# Import our model builder from model.py (same directory)
try:
    from model import build_model, print_model_summary
except ImportError:
    log.error("model.py not found in the same directory.")
    sys.exit(1)

# ── Configuration ─────────────────────────────────────────────────────────────
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))

# Input: preprocessed data created by preprocess.py
PREPROCESSED    = os.path.join(SCRIPT_DIR, "preprocessed_data.json")

# Output artefacts
MODEL_OUT       = os.path.join(SCRIPT_DIR, "chatbot_model_w8.keras")
TOKENIZER_OUT   = os.path.join(SCRIPT_DIR, "tokenizer_w8.pkl")
ENCODER_OUT     = os.path.join(SCRIPT_DIR, "label_encoder_w8.pkl")
HISTORY_CSV     = os.path.join(SCRIPT_DIR, "training_history.csv")
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "best_checkpoint.keras")

# Hyperparameters
MAX_WORDS   = 2000      # vocabulary ceiling for Keras Tokenizer
MAX_LEN     = 20        # maximum token sequence length (pad/truncate)
EPOCHS      = 200       # maximum training epochs (EarlyStopping may halt earlier)
BATCH_SIZE  = 16        # mini-batch size

# Train / validation / test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15      # = 1 - TRAIN_RATIO - VAL_RATIO

# ── Console colour helpers ─────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

def ok(msg):    log.info(f"{GREEN}✓ {msg}{RESET}")
def err(msg):   log.error(f"{RED}✗ {msg}{RESET}")
def info(msg):  log.info(f"{CYAN}ℹ {msg}{RESET}")
def warn(msg):  log.warning(f"{YELLOW}⚠ {msg}{RESET}")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_preprocessed(path: str) -> tuple:
    """
    Load the preprocessed data JSON produced by preprocess.py.

    Extracts the 'joined' lemmatised sentence (used as model input text) and
    the 'intent' label for each record.

    Args:
        path (str): Path to preprocessed_data.json

    Returns:
        tuple (sentences: list[str], labels: list[str])

    Raises:
        FileNotFoundError: if the JSON file is absent
        KeyError:          if expected fields are missing in the JSON
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Preprocessed data not found: {path}\n"
            "Run  python preprocess.py  first."
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records   = data.get("records", [])
    sentences = []
    labels    = []

    for rec in records:
        joined = rec.get("joined", "").strip()
        intent = rec.get("intent", "").strip()
        if joined and intent:
            sentences.append(joined)
            labels.append(intent)

    if not sentences:
        raise ValueError("No valid records found in preprocessed_data.json.")

    ok(f"Loaded {len(sentences)} records | {len(set(labels))} intent classes")
    return sentences, labels


# ── Tokenisation and sequence encoding ───────────────────────────────────────

def build_sequences(sentences: list, tokenizer=None) -> tuple:
    """
    Fit (or apply) a Keras Tokenizer and convert sentences to padded integer
    sequences.

    The Tokenizer assigns a unique integer index to each token in the
    vocabulary.  pad_sequences then makes all sequences the same fixed length
    (MAX_LEN) by zero-padding short sequences and truncating long ones.

    Args:
        sentences  (list[str]): Preprocessed sentences.
        tokenizer  (Tokenizer or None): If None, a new tokenizer is fitted.

    Returns:
        tuple (X: np.ndarray, tokenizer: Tokenizer, vocab_size: int)
            X          – shape (n_samples, MAX_LEN), padded integer sequences
            tokenizer  – fitted Tokenizer object
            vocab_size – effective vocabulary size
    """
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
        tokenizer.fit_on_texts(sentences)

    vocab_size = min(len(tokenizer.word_index) + 1, MAX_WORDS)
    sequences  = tokenizer.texts_to_sequences(sentences)
    X          = pad_sequences(sequences, maxlen=MAX_LEN, padding="post",
                               truncating="post")

    info(f"Vocabulary size: {vocab_size} | Sequence tensor shape: {X.shape}")
    return X, tokenizer, vocab_size


# ── Data splitting ────────────────────────────────────────────────────────────

def split_data(X: np.ndarray, y: np.ndarray, labels_raw: list) -> tuple:
    """
    Stratified train / validation / test split.

    Stratification ensures each intent class is proportionally represented in
    all three splits, preventing any class from being under-represented in the
    training set due to random chance.

    Split:
        70% Training   – gradient updates
        15% Validation – callback monitoring (EarlyStopping / ReduceLROnPlateau)
        15% Test       – final held-out evaluation (never seen during training)

    Args:
        X          (np.ndarray): Padded integer sequences.
        y          (np.ndarray): Integer-encoded intent labels.
        labels_raw (list[str]): Original string labels (for stratification).

    Returns:
        tuple: X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First split: (train+val) vs test
    X_tv, X_test, y_tv, y_test, lbl_tv, _ = train_test_split(
        X, y, labels_raw,
        test_size=TEST_RATIO,
        random_state=42,
        stratify=labels_raw,
    )

    # Second split: train vs val (from the train+val portion)
    val_proportion = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size=val_proportion,
        random_state=42,
        stratify=lbl_tv,
    )

    info(
        f"Data split → "
        f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── History CSV export ────────────────────────────────────────────────────────

def save_history_csv(history, path: str) -> None:
    """
    Save the Keras training history to a CSV file.

    Columns: epoch, loss, accuracy, val_loss, val_accuracy

    This structured format allows the training progress to be reviewed
    in Excel, pandas, or any spreadsheet application.

    Args:
        history: Keras History object returned by model.fit()
        path (str): Destination CSV file path
    """
    hist = history.history
    epochs = range(1, len(hist["loss"]) + 1)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "accuracy", "val_loss", "val_accuracy"])
        for ep, lo, ac, vl, va in zip(
            epochs,
            hist["loss"],
            hist["accuracy"],
            hist.get("val_loss", [None] * len(epochs)),
            hist.get("val_accuracy", [None] * len(epochs)),
        ):
            writer.writerow([
                ep,
                round(lo, 6),
                round(ac, 6),
                round(vl, 6) if vl is not None else "",
                round(va, 6) if va is not None else "",
            ])

    ok(f"Training history saved → {path}")


# ── Main training pipeline ────────────────────────────────────────────────────

def train():
    """
    Execute the full end-to-end training pipeline.

    Steps:
        1.  Load preprocessed data (preprocessed_data.json)
        2.  Encode intent labels with LabelEncoder
        3.  Tokenise sentences and build padded sequences
        4.  Stratified 70/15/15 train/val/test split
        5.  Build BiLSTM model (model.py)
        6.  Train with EarlyStopping + ReduceLROnPlateau + ModelCheckpoint
        7.  Evaluate on held-out test set
        8.  Save model, tokenizer, label encoder, and training history CSV
    """
    print(f"\n{CYAN}{'=' * 65}")
    print("  Hypotify Week 8 – BiLSTM Intent Classifier Training")
    print(f"{'=' * 65}{RESET}\n")

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    info("Step 1: Loading preprocessed data …")
    try:
        sentences, labels = load_preprocessed(PREPROCESSED)
    except (FileNotFoundError, ValueError) as exc:
        err(str(exc))
        sys.exit(1)

    # ── Step 2: Encode labels ─────────────────────────────────────────────────
    info("Step 2: Encoding intent labels …")
    le = LabelEncoder()
    y  = le.fit_transform(labels)
    n_classes = len(le.classes_)
    info(f"Intent classes ({n_classes}): {list(le.classes_)}")

    # ── Step 3: Tokenise and pad sequences ────────────────────────────────────
    info("Step 3: Tokenising and padding sequences …")
    X, tokenizer, vocab_size = build_sequences(sentences)

    # ── Step 4: Stratified train/val/test split ───────────────────────────────
    info("Step 4: Performing stratified 70/15/15 data split …")
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            X, y, labels
        )
    except ValueError as exc:
        # If any class has too few samples for stratification, fall back
        warn(f"Stratified split failed ({exc}); using random split.")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(VAL_RATIO + TEST_RATIO), random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

    # ── Step 5: Build model ───────────────────────────────────────────────────
    info("Step 5: Building BiLSTM model architecture …")
    model = build_model(vocab_size=vocab_size, n_classes=n_classes, max_len=MAX_LEN)
    print_model_summary(model, max_len=MAX_LEN)

    # ── Step 6: Train with callbacks ──────────────────────────────────────────
    info(f"Step 6: Training for up to {EPOCHS} epochs …")

    callbacks = [
        # Stop training when val_accuracy plateaus (patience=25 epochs)
        EarlyStopping(
            monitor="val_accuracy",
            patience=25,
            restore_best_weights=True,
            verbose=1,
        ),
        # Reduce learning rate when validation accuracy stagnates
        ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.5,
            patience=12,
            min_lr=1e-6,
            verbose=1,
        ),
        # Save the best model checkpoint during training
        ModelCheckpoint(
            filepath=CHECKPOINT_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=0,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    epochs_run  = len(history.history["accuracy"])
    final_acc   = history.history["accuracy"][-1]
    final_vacc  = history.history.get("val_accuracy", [0])[-1]
    ok(f"Training complete — {epochs_run} epochs | "
       f"train acc: {final_acc:.4f} | val acc: {final_vacc:.4f}")

    # ── Step 7: Evaluate on test set ──────────────────────────────────────────
    info("Step 7: Evaluating on held-out test set …")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    ok(f"Test set accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

    # Per-class classification report for insight into per-intent performance
    y_pred      = np.argmax(model.predict(X_test, verbose=0), axis=1)
    target_names = list(le.classes_)
    print("\n  --- Per-Class Classification Report ---")
    try:
        report = classification_report(y_test, y_pred, target_names=target_names,
                                       zero_division=0)
        print(report)
    except Exception as exc:
        warn(f"Could not print classification report: {exc}")

    # ── Step 8: Save all artefacts ────────────────────────────────────────────
    info("Step 8: Saving model artefacts …")

    # 8a: Trained model
    model.save(MODEL_OUT)
    ok(f"Model saved      → {MODEL_OUT}")

    # 8b: Tokenizer
    with open(TOKENIZER_OUT, "wb") as f:
        pickle.dump(tokenizer, f)
    ok(f"Tokenizer saved  → {TOKENIZER_OUT}")

    # 8c: Label encoder
    with open(ENCODER_OUT, "wb") as f:
        pickle.dump(le, f)
    ok(f"Label encoder    → {ENCODER_OUT}")

    # 8d: Training history CSV
    save_history_csv(history, HISTORY_CSV)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{GREEN}{'=' * 65}")
    print("  TRAINING COMPLETE")
    print(f"{'=' * 65}{RESET}")
    print(f"  Final train accuracy : {final_acc:.4f}")
    print(f"  Final val accuracy   : {final_vacc:.4f}")
    print(f"  Test accuracy        : {test_acc:.4f}")
    print(f"  Epochs run           : {epochs_run}")
    print(f"\n  Saved artefacts:")
    print(f"    {MODEL_OUT}")
    print(f"    {TOKENIZER_OUT}")
    print(f"    {ENCODER_OUT}")
    print(f"    {HISTORY_CSV}")
    print(f"{'=' * 65}{RESET}\n")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print(f"\n{YELLOW}  Training interrupted by user.{RESET}\n")
        sys.exit(0)
    except Exception as exc:
        err(f"Unexpected error during training: {exc}")
        log.exception(exc)
        sys.exit(1)
