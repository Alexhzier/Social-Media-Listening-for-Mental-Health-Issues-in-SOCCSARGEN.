"""
CNN-LSTM architecture for multi-class distress-pattern classification.

Outputs one of 5 categories: Depression, Stress, Anxiety, Addiction,
Suicidal Behavior. Designed for Cebuano, Tagalog, and English.

ETHICAL NOTE: This model classifies LINGUISTIC PATTERNS only. It does NOT
diagnose mental illness and must not be used as a substitute for professional
assessment or treatment. Research and screening support only.
"""

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    Embedding,
    LSTM,
)
from tensorflow.keras.optimizers import Adam


NUM_CLASSES = 5  # 5 categories: Depression, Stress, Anxiety, Addiction, Suicidal Behavior


def build_cnn_lstm(
    vocab_size: int,
    num_classes: int = NUM_CLASSES,
    embedding_dim: int = 128,
    max_sequence_length: int = 128,
    num_filters: int = 96,
    kernel_size: int = 3,
    lstm_units: int = 64,
    dropout_rate: float = 0.4,
    learning_rate: float = 3e-4,
) -> Model:
    """
    Build CNN-LSTM for multi-class classification (5 categories).

    Pipeline:
      1. Embedding: map token ids to dense vectors.
      2. CNN: extract local n-gram features.
      3. BiLSTM: capture contextual dependencies in both directions.
      4. Dense + softmax: probability distribution over categories.
    """
    embedding_input_dim = vocab_size + 1

    inputs = Input(shape=(max_sequence_length,), dtype="int32")

    x = Embedding(
        input_dim=embedding_input_dim,
        output_dim=embedding_dim,
        mask_zero=False,
    )(inputs)

    x = Conv1D(
        filters=num_filters,
        kernel_size=kernel_size,
        activation="relu",
        padding="same",
    )(x)
    x = Dropout(dropout_rate)(x)

    x = Bidirectional(LSTM(lstm_units, return_sequences=False))(x)
    x = Dropout(dropout_rate)(x)

    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
