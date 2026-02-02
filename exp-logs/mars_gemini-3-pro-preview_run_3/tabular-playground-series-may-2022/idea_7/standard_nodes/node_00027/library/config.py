import os
import torch


class Config:
    """
    Central configuration for the Dual-Stream Gated Funnel Network strategy.
    Defines file paths, hyperparameters, and feature schemas.
    """

    # =========================================================================
    # Directories and Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_7"
    SUBMISSION_DIR = "./submission"

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Data Paths (using metadata splits for training flow)
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Original Test Path (for reference or raw loading if needed)
    RAW_TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Paths
    CACHE_TRAIN_PATH = os.path.join(WORKING_DIR, "train_processed.parquet")
    CACHE_VAL_PATH = os.path.join(WORKING_DIR, "val_processed.parquet")
    CACHE_TEST_PATH = os.path.join(WORKING_DIR, "test_processed.parquet")
    CACHE_ENCODER_PATH = os.path.join(WORKING_DIR, "ordinal_encoder.npy")
    CACHE_SCALER_PATH = os.path.join(WORKING_DIR, "scaler.npy")

    # =========================================================================
    # Global Settings
    # =========================================================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # For DataLoader

    # =========================================================================
    # Data Definitions
    # =========================================================================
    ID_COL = "id"
    TARGET_COL = "target"
    SOURCE_PATH_COL = "source_path"

    # Feature Engineering Constants
    F_27_COL = "f_27"
    F_27_SEQ_LEN = 10  # Length of the string in f_27
    UNIQUE_CHAR_COUNT_COL = "unique_character_count"

    # Continuous Features: f_00 to f_28 (excluding f_27) + unique_character_count
    # f_00 to f_26 are continuous. f_28 is continuous.
    CONT_FEATURES = [f"f_{i:02d}" for i in range(27)] + ["f_28", UNIQUE_CHAR_COUNT_COL]

    # Categorical Features: f_29, f_30 + decomposed f_27 characters
    # We will name decomposed columns f_27_0, f_27_1, ...
    CAT_FEATURES_BASE = ["f_29", "f_30"]
    CAT_FEATURES_F27 = [f"f_27_{i}" for i in range(F_27_SEQ_LEN)]
    CAT_FEATURES = CAT_FEATURES_BASE + CAT_FEATURES_F27

    # Dimensions
    NUM_CONT_FEATURES = len(CONT_FEATURES)
    NUM_CAT_FEATURES = len(CAT_FEATURES)

    # =========================================================================
    # Model Hyperparameters (Dual-Stream Gated Funnel Network)
    # =========================================================================
    EMBEDDING_DIM = 32
    HIDDEN_LAYERS = [512, 256, 128]
    DROPOUT_RATE = 0.25
    OUTPUT_DIM = 1

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 1024
    EPOCHS = 30
    PATIENCE = 5  # For Early Stopping

    # Optimizer (AdamW)
    LEARNING_RATE = 1e-2  # Max LR for OneCycle
    WEIGHT_DECAY = 1e-5

    # Scheduler (OneCycleLR)
    PCT_START = 0.3
    DIV_FACTOR = 25.0
    FINAL_DIV_FACTOR = 1000.0

    # =========================================================================
    # Debugging / Development
    # =========================================================================
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 5000  # Number of samples to use in debug mode

    def __init__(self):
        pass

    @classmethod
    def print_config(cls):
        print("=" * 40)
        print(f"CONFIG: {cls.__name__}")
        print("=" * 40)
        print(f"Device: {cls.DEVICE}")
        print(f"Seed: {cls.SEED}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Epochs: {cls.EPOCHS}")
        print(f"Embedding Dim: {cls.EMBEDDING_DIM}")
        print(f"Hidden Layers: {cls.HIDDEN_LAYERS}")
        print(f"Dropout: {cls.DROPOUT_RATE}")
        print(f"Weight Decay: {cls.WEIGHT_DECAY}")
        print(f"Working Dir: {cls.WORKING_DIR}")
        print("=" * 40)
