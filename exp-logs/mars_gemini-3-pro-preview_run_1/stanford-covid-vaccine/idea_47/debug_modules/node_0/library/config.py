import os
import random
import numpy as np
import torch


class Config:
    """
    Configuration class for the Context-Modulated Wide-Stream Residual BiGRU model.
    Centralizes all hyperparameters, file paths, and reproducibility settings.
    """

    # =========================================================================
    # Directories and Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_47"
    SUBMISSION_DIR = "./submission"

    # Ensure necessary writeable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Data Paths (using pre-generated metadata)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Specifications
    # =========================================================================
    SEQ_LEN = 107
    PRED_LEN = 68

    # The 3 scored targets (filtering out deg_pH10 and deg_50C per strategy)
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    N_OUTPUTS = len(TARGET_COLS)

    # =========================================================================
    # Model Architecture Hyperparameters
    # =========================================================================
    # Embeddings
    EMBED_DIM_SEQ = 128  # Atomic Sequence
    EMBED_DIM_LOOP = 64  # Predicted Loop Type
    EMBED_DIM_DIST = 64  # Signed Sinusoidal Pairing Distance

    # Backbone (Wide-Stream Residual BiGRU)
    HIDDEN_DIM = 384  # Full residual stream width
    N_LAYERS = 6  # Depth of the backbone
    DROPOUT = 0.2  # Inter-layer dropout (p=0.2)

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    BATCH_SIZE = 64  # Adjusted for A100 GPU
    NUM_EPOCHS = 20  # Fixed budget
    LR = 1e-3  # AdamW Learning Rate
    WEIGHT_DECAY = 1e-4  # Low weight decay to preserve recurrent signals
    CLIP_GRAD = 1.0  # Gradient clipping norm

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # For DataLoader

    # =========================================================================
    # Utilities
    # =========================================================================
    @staticmethod
    def set_seed(seed=42):
        """Sets fixed random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Ensure deterministic behavior in cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Set python hash seed
        os.environ["PYTHONHASHSEED"] = str(seed)
