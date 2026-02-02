import os
import torch


class Config:
    """
    Configuration class for the Dual-Context CodeBERT Network (DC-CodeBERT) project.
    Contains file paths, hyperparameters, and global settings.
    """

    # ==========================================
    # 1. Global Settings & Reproducibility
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 1000  # Number of samples to use in debug mode

    # ==========================================
    # 2. File System Paths
    # ==========================================
    # Base directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_4"
    SUBMISSION_DIR = "./submission"

    # Ensure writable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Paths (Parquet files for processed features)
    TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

    # Model Checkpoint Path
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # ==========================================
    # 3. Model Hyperparameters
    # ==========================================
    MODEL_NAME = "microsoft/codebert-base"

    # Dimensions
    HIDDEN_DIM = 768  # CodeBERT hidden size
    LATENT_DIM = 512  # Shared latent space dimension for projection heads

    # Sequence Processing
    MAX_LEN = 512  # Max sequence length for tokenization

    # ==========================================
    # 4. Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 5

    # Optimization
    WEIGHT_DECAY = 0.01
    DROPOUT = 0.1

    # ==========================================
    # 5. Compute & Resources
    # ==========================================
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print("=" * 30)
        print("CONFIG")
        print("=" * 30)
        for k, v in cls.__dict__.items():
            if not k.startswith("__") and not callable(v):
                print(f"{k}: {v}")
        print("=" * 30)
