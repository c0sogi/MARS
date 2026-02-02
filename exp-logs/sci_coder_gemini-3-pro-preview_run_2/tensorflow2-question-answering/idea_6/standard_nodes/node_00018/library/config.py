import os
import torch


class Config:
    # --- Reproducibility ---
    SEED = 42

    # --- Paths ---
    # Input Data
    INPUT_DIR = "./input"
    TRAIN_FILE = os.path.join(INPUT_DIR, "simplified-nq-train.jsonl")
    TEST_FILE = os.path.join(INPUT_DIR, "simplified-nq-test.jsonl")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata
    METADATA_DIR = "./metadata"
    TRAIN_META = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META = os.path.join(METADATA_DIR, "validation_metadata.csv")
    TEST_META = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Working Directory (Cache & Outputs)
    WORKING_DIR = "./working/idea_6"
    CACHE_DIR = WORKING_DIR  # Data cache
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # --- Data Preprocessing ---
    # Sequence Lengths
    MAX_Q_LEN = 20  # N: Fixed length for questions
    MAX_C_LEN = 300  # M: Fixed length for long answer candidates

    # Vocabulary
    VOCAB_SIZE = 20000  # Max vocab size
    MIN_FREQ = 2  # Minimum frequency for a token to be included
    UNK_TOKEN = "<UNK>"
    PAD_TOKEN = "<PAD>"

    # Embedding
    EMBED_DIM = 100  # Dimension for word embeddings
    USE_PRETRAINED = False  # If True, would load GloVe, else train from scratch/random

    # Caching filenames
    VOCAB_CACHE_FILE = os.path.join(CACHE_DIR, "vocab.npy")
    EMBED_MATRIX_CACHE_FILE = os.path.join(CACHE_DIR, "embedding_matrix.npy")
    TRAIN_FEATURES_CACHE = os.path.join(CACHE_DIR, "train_features.parquet")
    VAL_FEATURES_CACHE = os.path.join(CACHE_DIR, "val_features.parquet")
    TEST_FEATURES_CACHE = os.path.join(CACHE_DIR, "test_features.parquet")

    # --- Model Architecture ---
    # CNN Encoder
    CNN_FILTERS = 64
    CNN_KERNEL_SIZE = (3, 3)
    CNN_POOL_SIZE = (2, 2)

    # Span Prediction (1D CNN)
    SPAN_CNN_FILTERS = 32
    SPAN_CNN_KERNEL_SIZE = 5

    # Heads
    HIDDEN_DIM = 128
    DROPOUT_RATE = 0.3
    NUM_YN_CLASSES = 3  # YES, NO, NONE

    # --- Training ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 128
    NUM_EPOCHS = 5
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5

    # Loss Weights for Multi-Task Learning
    LOSS_WEIGHT_RANK = 1.0
    LOSS_WEIGHT_SPAN = 0.5
    LOSS_WEIGHT_YN = 0.5

    # Sampling
    NEGATIVE_RATIO = (
        1  # Number of negative candidates per positive candidate in training
    )

    # Optimization
    EARLY_STOPPING_PATIENCE = 2

    # Inference Thresholds
    LONG_ANSWER_THRESHOLD = 0.5  # Confidence score to predict a long answer

    # Debugging
    DEBUG_SAMPLE_SIZE = (
        0  # Set to > 0 (e.g., 1000) to limit dataset size for quick debugging
    )
