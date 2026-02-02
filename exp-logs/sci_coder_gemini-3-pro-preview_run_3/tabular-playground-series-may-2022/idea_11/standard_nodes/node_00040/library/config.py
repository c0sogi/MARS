import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """Sets the random seed for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class Config:
    # ==========================================
    # Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_11"
    SUBMISSION_DIR = "./submission"

    # Input Files (using metadata for split consistency)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    CACHE_PATH_TRAIN = os.path.join(WORKING_DIR, "train_processed.parquet")
    CACHE_PATH_VAL = os.path.join(WORKING_DIR, "val_processed.parquet")
    CACHE_PATH_TEST = os.path.join(WORKING_DIR, "test_processed.parquet")
    METADATA_CACHE_PATH = os.path.join(
        WORKING_DIR, "metadata.npy"
    )  # For vocab sizes, scalers, etc.

    # ==========================================
    # Data & Feature Engineering
    # ==========================================
    SEED = 42
    NUM_WORKERS = 4

    # Feature Definitions
    # f_27 is decomposed into 10 characters
    F_27_LENGTH = 10

    # Categorical Features: f_29, f_30 + 10 chars from f_27
    # We will handle f_29 and f_30 as categorical as per strategy
    CATEGORICAL_COLS = ["f_29", "f_30"]

    # Continuous Features: f_00 to f_28 (excluding f_27) + unique_character_count
    # Note: f_28 is continuous. f_29, f_30 are categorical.
    CONTINUOUS_COLS = [f"f_{i:02d}" for i in range(29) if i != 27] + [
        "unique_character_count"
    ]

    # ==========================================
    # Model Hyperparameters (Layer-Normalized Funnel MLP)
    # ==========================================
    EMBEDDING_DIM = 16
    HIDDEN_LAYERS = [512, 256, 128]
    TOKEN_DROPOUT_RATE = 0.1  # For embeddings
    DROPOUT_RATE = 0.1  # For dense blocks
    USE_LAYER_NORM = True

    # ==========================================
    # Training & Optimization
    # ==========================================
    BATCH_SIZE = 1024
    EPOCHS = 30
    LEARNING_RATE = 1e-2  # Max LR for OneCycle
    WEIGHT_DECAY = 1e-5
    PATIENCE = 5  # Early stopping patience

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """Creates necessary directories and sets random seeds."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        set_seed(cls.SEED)

        # Print configuration summary
        print(f"Configuration Setup Complete.")
        print(f"Device: {cls.DEVICE}")
        print(f"Working Directory: {cls.WORKING_DIR}")
        print(f"Batch Size: {cls.BATCH_SIZE}, LR: {cls.LEARNING_RATE}")
