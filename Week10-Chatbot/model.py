"""
model.py
========
Week 8 Assignment – Chatbot Model Development and Training
ITEC5025 – Hypotify Clinical Chatbot
Author: Shruti Malik
Date:   2026-02-28

Purpose:
    Define the advanced neural network architecture for intent classification.
    This module is imported by train.py and chatbot.py.

Architecture (Bidirectional LSTM):
    Embedding(vocab_size, 128, input_length=MAX_LEN)
    → SpatialDropout1D(0.2)
    → Bidirectional(LSTM(64, return_sequences=True))
    → GlobalMaxPooling1D()
    → Dense(128, relu)
    → Dropout(0.4)
    → Dense(64, relu)
    → Dense(n_classes, softmax)

Compared to Week 7's simple Embedding → GlobalAveragePooling1D → Dense design,
Week 8 adds:
    - Bidirectional LSTM: captures word-order context in both forward and
      backward directions, giving the model a sense of sentence structure.
    - SpatialDropout1D: drops entire feature maps in the embedding layer
      during training, a regularisation technique specific to sequence models.
    - GlobalMaxPooling1D: extracts the strongest (max) feature across the
      time dimension rather than averaging, preserving discriminative signals.

Usage:
    from model import build_model, MODEL_CONFIG
"""

import sys
import logging

log = logging.getLogger(__name__)

# ── TensorFlow imports ────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Embedding,
        SpatialDropout1D,
        Bidirectional,
        LSTM,
        GlobalMaxPooling1D,
        Dense,
        Dropout,
        BatchNormalization,
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2
except ImportError:
    log.error("TensorFlow not installed. Run: pip install tensorflow")
    sys.exit(1)

# ── Model configuration constants ─────────────────────────────────────────────
MODEL_CONFIG = {
    "embedding_dim":      128,     # size of each word embedding vector
    "lstm_units":         64,      # number of LSTM memory units per direction
    "dense_1_units":      128,     # units in the first dense hidden layer
    "dense_2_units":      64,      # units in the second dense hidden layer
    "spatial_dropout":    0.2,     # SpatialDropout rate for embedding layer
    "dense_dropout":      0.4,     # Dropout rate for dense layers
    "l2_reg":             0.001,   # L2 regularisation factor
    "learning_rate":      0.001,   # Adam optimizer learning rate
}


def build_model(vocab_size: int, n_classes: int, max_len: int) -> tf.keras.Model:
    """
    Construct and compile the Bidirectional LSTM intent classifier.

    The architecture is designed with clinical NLP in mind:
        - Medical queries often use complex multi-word phrases (e.g., "show me
          the age distribution") where word order matters, making LSTM superior
          to simple bag-of-words averaging (Week 7's GlobalAveragePooling1D).
        - Bidirectionality allows the model to read queries both left-to-right
          and right-to-left, capturing dependencies in either direction.
        - BatchNormalization after dense layers stabilises and accelerates
          training by normalising layer activations.

    Args:
        vocab_size (int): Size of the vocabulary (number of unique tokens + 1).
        n_classes  (int): Number of intent classes to predict.
        max_len    (int): Padded sequence length (must match train.py MAX_LEN).

    Returns:
        tf.keras.Sequential: A compiled Keras model ready for training.
    """
    cfg = MODEL_CONFIG

    model = Sequential(
        [
            # ── Layer 1: Embedding ─────────────────────────────────────────────
            # Converts integer token indices into dense embedding vectors.
            # Trained jointly with the rest of the network (no pre-trained
            # weights), so embeddings are tuned to our clinical intent domain.
            Embedding(
                input_dim=vocab_size,
                output_dim=cfg["embedding_dim"],
                input_length=max_len,
                name="embedding",
            ),

            # ── Layer 2: SpatialDropout1D ──────────────────────────────────────
            # Randomly zeros out entire embedding feature maps (columns) during
            # training. This prevents individual embedding dimensions from
            # co-adapting, acting as feature-level regularisation.
            SpatialDropout1D(cfg["spatial_dropout"], name="spatial_dropout"),

            # ── Layer 3: Bidirectional LSTM ────────────────────────────────────
            # Wraps two LSTM layers: one reads the sequence left-to-right,
            # the other right-to-left. Their outputs are concatenated.
            # return_sequences=True passes the full sequence of hidden states
            # to the next layer (required for GlobalMaxPooling1D).
            Bidirectional(
                LSTM(
                    cfg["lstm_units"],
                    return_sequences=True,
                    dropout=0.2,              # input-gate dropout
                    recurrent_dropout=0.1,    # recurrent-state dropout
                    kernel_regularizer=l2(cfg["l2_reg"]),
                ),
                name="bidirectional_lstm",
            ),

            # ── Layer 4: GlobalMaxPooling1D ────────────────────────────────────
            # Takes the maximum value across the time dimension for each
            # feature. This collapses the variable-length sequence into a
            # fixed-length vector, keeping the strongest activation per feature.
            GlobalMaxPooling1D(name="global_max_pool"),

            # ── Layer 5: Dense hidden layer 1 ─────────────────────────────────
            # Learns higher-level combinations of LSTM features.
            Dense(
                cfg["dense_1_units"],
                activation="relu",
                kernel_regularizer=l2(cfg["l2_reg"]),
                name="dense_1",
            ),
            BatchNormalization(name="batch_norm_1"),
            Dropout(cfg["dense_dropout"], name="dropout_1"),

            # ── Layer 6: Dense hidden layer 2 ─────────────────────────────────
            # Secondary feature compression for intent discrimination.
            Dense(
                cfg["dense_2_units"],
                activation="relu",
                kernel_regularizer=l2(cfg["l2_reg"]),
                name="dense_2",
            ),
            BatchNormalization(name="batch_norm_2"),
            Dropout(cfg["dense_dropout"] * 0.5, name="dropout_2"),

            # ── Layer 7: Output layer ──────────────────────────────────────────
            # Softmax outputs a probability distribution over all intent classes.
            # The predicted intent is the class with the highest probability.
            Dense(n_classes, activation="softmax", name="output"),
        ],
        name="hypotify_w8_bilstm_classifier",
    )

    # ── Compile ────────────────────────────────────────────────────────────────
    # Adam (Adaptive Moment Estimation) adjusts learning rates per parameter,
    # making it well-suited for sparse gradient problems like NLP.
    model.compile(
        optimizer=Adam(learning_rate=cfg["learning_rate"]),
        loss="sparse_categorical_crossentropy",  # works with integer class labels
        metrics=["accuracy"],
    )

    return model


def print_model_summary(model: tf.keras.Model, max_len: int = 20) -> None:
    """
    Print a formatted summary of the model architecture to stdout.

    Builds the model with a dummy input shape if it has not been built yet
    (required by Keras 3+ before count_params() can be called).

    Args:
        model  (tf.keras.Model): A compiled Keras model.
        max_len (int): Sequence length used during training (for build shape).
    """
    print("\n" + "=" * 65)
    print("  Model Architecture Summary")
    print("=" * 65)
    # Keras 3 requires an explicit build / forward pass before count_params()
    try:
        model.build(input_shape=(None, max_len))
    except Exception:
        pass  # model may already be built after training
    model.summary()
    try:
        total_params = model.count_params()
        print(f"\n  Total trainable parameters: {total_params:,}")
    except Exception:
        pass  # skip param count if model is still unbuilt
    print("=" * 65 + "\n")


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Quick architecture sanity check – builds a model with dummy dimensions
    and prints the summary without requiring real training data.
    """
    print("\nBuilding sample model for architecture verification …\n")
    try:
        sample_model = build_model(vocab_size=500, n_classes=15, max_len=20)
        print_model_summary(sample_model)
        print("  Model architecture defined successfully!")
    except Exception as exc:
        print(f"  ERROR: {exc}")
        sys.exit(1)
