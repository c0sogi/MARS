import os
import torch


class Config:
    """
    Configuration class for the Independent-Projection Parallel Funnel Ensemble (IPPFE).
    Centralizes all file paths, model hyperparameters, and training settings.
    """

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_28"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Cache directory for processed data (Parquet/NPY)
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Data Paths (using metadata splits)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # =========================================================================
    # Global Settings
    # =========================================================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Number of workers for DataLoaders

    # =========================================================================
    # Data Processing Parameters
    # =========================================================================
    TARGET_COL = "target"
    ID_COL = "id"

    # Feature Engineering
    # f_27 is decomposed into a sequence of characters
    F27_SEQ_LEN = 10

    # Debugging / Development
    # Set MAX_SAMPLES to an integer (e.g., 5000) to restrict dataset size for rapid testing/debugging.
    # Set to None to use the full dataset for final training.
    MAX_SAMPLES = None

    # =========================================================================
    # Model Architecture: IPPFE
    # =========================================================================
    # Independent-Projection Parallel Funnel Ensemble
    # The model consists of 5 independent streams within a single graph.

    # Categorical Embedding Dimension (applied independently per stream)
    EMBEDDING_DIM = 16

    # Stream Configurations
    # Defines the backbone topology (Funnel MLP) and regularization for each stream.
    # Format: List of dictionaries containing 'hidden_dims' and 'dropout'.
    STREAMS = [
        # Stream 1: Standard Funnel, Dropout 0.20 (Anchor)
        {"hidden_dims": [512, 256, 128], "dropout": 0.20},
        # Stream 2: Standard Funnel, Dropout 0.20 (Anchor)
        {"hidden_dims": [512, 256, 128], "dropout": 0.20},
        # Stream 3: Wide Funnel, Dropout 0.25 (Capacity Variant)
        {"hidden_dims": [1024, 512, 256], "dropout": 0.25},
        # Stream 4: Standard Funnel, Dropout 0.15 (Safe-Aggressive)
        {"hidden_dims": [512, 256, 128], "dropout": 0.15},
        # Stream 5: Standard Funnel, Dropout 0.30 (Conservative)
        {"hidden_dims": [512, 256, 128], "dropout": 0.30},
    ]

    # Activation Function used in MLPs
    ACTIVATION = "ReLU"

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 1024
    EPOCHS = 50

    # Optimization (AdamW)
    OPTIMIZER_NAME = "AdamW"
    WEIGHT_DECAY = 1e-4

    # Scheduler (OneCycleLR)
    SCHEDULER_NAME = "OneCycleLR"
    MAX_LR = 1e-2
    PCT_START = 0.3  # Percentage of training to increase LR
