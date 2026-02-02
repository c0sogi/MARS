import os
import torch


class Config:
    # ==========================================
    # Paths
    # ==========================================
    # Input Metadata (Generated in previous steps)
    TRAIN_METADATA_PATH = "./metadata/train_metadata.csv"
    VAL_METADATA_PATH = "./metadata/val_metadata.csv"
    TEST_METADATA_PATH = "./metadata/test_metadata.csv"
    SAMPLE_SUBMISSION_PATH = "./input/sample_submission.csv"

    # Working Directories
    # Used for caching intermediate data (embeddings, features) and saving models
    WORKING_DIR = "./working/idea_3/"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    OUTPUT_DIR = "./submission/"

    # Model Artifacts
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_SAVE_PATH = os.path.join(OUTPUT_DIR, "submission.csv")

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    MODEL_NAME = "microsoft/deberta-v3-small"
    MAX_LENGTH = 1024  # Max sequence length for tokenizer
    HIDDEN_DROPOUT_PROB = 0.1

    # The dimension of the injected scalar feature vector
    # Features:
    # 1. char_len_diff
    # 2. word_len_diff
    # 3. char_len_ratio
    # 4. word_len_ratio
    # 5. newline_diff
    NUM_SCALAR_FEATURES = 5

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    TRAIN_BATCH_SIZE = 8  # Adjust based on VRAM (A100 40GB can handle more, but 8 is safe for stability)
    VALID_BATCH_SIZE = 16
    NUM_EPOCHS = 3

    # Differential Learning Rates
    LR_BACKBONE = 2e-5  # Lower LR for pre-trained transformer
    LR_HEAD = 1e-3  # Higher LR for the new regression/classification head

    WEIGHT_DECAY = 0.01
    PATIENCE = 2  # Early stopping patience

    # Device configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # For DataLoader

    # ==========================================
    # Feature Engineering Settings
    # ==========================================
    # List of scalar features to generate and inject
    SCALAR_FEATURE_LIST = [
        "char_len_diff",  # len(A) - len(B) (chars)
        "word_len_diff",  # len(A) - len(B) (words)
        "char_len_ratio",  # len(A) / (len(B) + epsilon)
        "word_len_ratio",  # len(A) / (len(B) + epsilon)
        "newline_diff",  # count(\n in A) - count(\n in B)
    ]

    @staticmethod
    def setup():
        """Ensure necessary directories exist."""
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
