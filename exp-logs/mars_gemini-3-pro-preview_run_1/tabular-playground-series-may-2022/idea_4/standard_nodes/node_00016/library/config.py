import os
import torch


class Config:
    """
    Centralized configuration for the Positional-Aware Hybrid Transformer pipeline.
    Includes file paths, data parameters, model architecture specifications,
    and training hyperparameters.
    """

    # -------------------------------------------------------------------------
    # 1. File Paths & Directories
    # -------------------------------------------------------------------------
    # Base directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_4"  # For caching intermediate files
    SUBMISSION_DIR = "./submission"

    # Ensure necessary writeable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Data paths (using metadata splits)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # 2. Data Configuration
    # -------------------------------------------------------------------------
    SEED = 42
    NUM_WORKERS = 4  # Number of DataLoader workers

    # Debugging: Set to an integer (e.g., 5000) to train on a subset, or None for full data
    DEBUG_SAMPLE_SIZE = None

    # Feature Definitions
    TARGET_COL = "target"
    SEQ_FEATURE = "f_27"
    # Numerical features: f_00 to f_30, excluding f_27
    NUM_FEATURES = [f"f_{i:02d}" for i in range(31) if i != 27]

    # Sequence Processing
    # f_27 is a string of fixed length 10
    MAX_SEQ_LEN = 10
    # Estimated vocabulary size (A-Z + special tokens).
    # Exact size will be determined during tokenization, but this serves as a default/max.
    VOCAB_SIZE = 40

    # Caching
    # If True, tries to load preprocessed tensors from WORKING_DIR
    LOAD_CACHED_DATA = True

    # -------------------------------------------------------------------------
    # 3. Model Architecture
    # -------------------------------------------------------------------------
    # Sequence Branch
    EMBED_DIM = 64
    TRANSFORMER_LAYERS = 2
    TRANSFORMER_HEADS = 4
    TRANSFORMER_FF_DIM = 128
    TRANSFORMER_DROPOUT = 0.1

    # Fusion & Classification Head (High-Capacity MLP)
    # The output of the transformer (pooled) and numerical features are concatenated
    # and passed through these layers.
    MLP_HIDDEN_DIMS = [512, 256, 128]
    MLP_DROPOUT = 0.1

    # -------------------------------------------------------------------------
    # 4. Training Configuration
    # -------------------------------------------------------------------------
    BATCH_SIZE = 1024
    EPOCHS = 30  # Extended epochs for OneCycleLR

    # Optimizer (AdamW)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler (OneCycleLR)
    # These parameters control the annealing curve
    PCT_START = 0.3
    DIV_FACTOR = 25.0
    FINAL_DIV_FACTOR = 1000.0

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
