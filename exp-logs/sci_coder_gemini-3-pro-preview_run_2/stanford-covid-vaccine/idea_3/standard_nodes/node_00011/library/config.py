import os
import torch
import numpy as np
import random


class Config:
    # =========================================================================
    # Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_3"
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Data Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Files
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    CACHE_DIR = os.path.join(WORKING_DIR, "data_cache")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # =========================================================================
    # Data Specifications
    # =========================================================================
    SEQ_LEN = 107
    PRED_LEN = 68

    # Vocabularies
    # Sequence: A, G, U, C
    VOCAB_SEQ = "AGUC"
    VOCAB_SIZE_SEQ = len(VOCAB_SEQ)
    TOKEN2INT_SEQ = {c: i for i, c in enumerate(VOCAB_SEQ)}

    # Structure: (, ), .
    VOCAB_STRUCT = "()."
    VOCAB_SIZE_STRUCT = len(VOCAB_STRUCT)
    TOKEN2INT_STRUCT = {c: i for i, c in enumerate(VOCAB_STRUCT)}

    # Loop Type: S, M, I, B, H, E, X
    VOCAB_LOOP = "SMIBHEX"
    VOCAB_SIZE_LOOP = len(VOCAB_LOOP)
    TOKEN2INT_LOOP = {c: i for i, c in enumerate(VOCAB_LOOP)}

    # Targets
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    NUM_TARGETS = len(TARGET_COLS)

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    EMBED_DIM = 32  # Dimension for learnable embeddings (Unused in optimized model)
    HIDDEN_DIM = 256  # Channel dimension for CNN/RNN
    KERNEL_SIZE = 3  # Convolution kernel size
    DILATIONS = [1, 2, 4, 8, 16]  # Dilated ResNet configuration
    DROPOUT = 0.3  # Dropout rate

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    EPOCHS = 30
    EARLY_STOPPING_PATIENCE = 7
    NUM_WORKERS = 4  # Number of DataLoader workers

    # Debugging flags to speed up development if needed
    DEBUG = False
    DEBUG_SAMPLES = 100

    # Device configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def set_seed(seed=42):
        """Sets the seed for reproducibility."""
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
