import os
import torch


class Config:
    # =========================================================================
    # Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_14"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = "./submission"

    # Raw Data
    TRAIN_DATA_PATH = os.path.join(INPUT_DIR, "simplified-nq-train.jsonl")
    TEST_DATA_PATH = os.path.join(INPUT_DIR, "simplified-nq-test.jsonl")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "validation_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Cache Files (Parquet/Numpy)
    VOCAB_PATH = os.path.join(CACHE_DIR, "vocab.npy")
    EMBEDDING_MATRIX_PATH = os.path.join(CACHE_DIR, "embedding_matrix.npy")
    TRAIN_FEATURES_PATH = os.path.join(CACHE_DIR, "train_features.parquet")
    VAL_FEATURES_PATH = os.path.join(CACHE_DIR, "val_features.parquet")
    TEST_FEATURES_PATH = os.path.join(CACHE_DIR, "test_features.parquet")

    # Model Checkpoint
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Output
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Processing Hyperparameters
    # =========================================================================
    MAX_SEQ_LEN = 300  # Maximum length for long answer candidate tokens
    MAX_Q_LEN = 64  # Maximum length for question tokens
    VOCAB_SIZE = 40000  # Maximum size of vocabulary
    UNK_TOKEN = "<UNK>"
    PAD_TOKEN = "<PAD>"

    # Debugging / Sampling
    # Set to None to use full dataset, or an integer (e.g., 1000) for debugging
    DEBUG_SAMPLE_SIZE = None

    # =========================================================================
    # Model Architecture Hyperparameters
    # =========================================================================
    EMBED_DIM = 100  # Dimension of word embeddings (e.g., matching GloVe)
    HIDDEN_DIM = 128  # Hidden dimension for Bi-GRU
    NUM_LAYERS = 1  # Number of GRU layers
    DROPOUT = 0.2  # Dropout rate
    NUM_CLASSES_YES_NO = 3  # YES, NO, NONE

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 5
    EARLY_STOPPING_PATIENCE = 2

    # Loss Weights for Multi-Task Learning
    WEIGHT_LONG_ANSWER = 1.0
    WEIGHT_SHORT_SPAN = 1.0
    WEIGHT_YES_NO = 0.5

    # Negative Sampling
    # Ratio of negative samples (no long answer) to positive samples in a batch
    NEGATIVE_SAMPLING_RATIO = 1.0

    # =========================================================================
    # Inference Hyperparameters
    # =========================================================================
    LONG_ANSWER_THRESHOLD = 0.5  # Threshold for binary classification of long answer

    # =========================================================================
    # System Settings
    # =========================================================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # For data loading

    @classmethod
    def setup(cls):
        """Ensures necessary directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set deterministic behavior for reproducibility
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
