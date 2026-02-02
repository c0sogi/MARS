import os
import torch


class Config:
    """
    Configuration class for the Channel-Attentive Distance-Aware Residual BiGRU strategy.
    Centralizes all hyperparameters, file paths, and environment settings.
    """

    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Working directory for Idea 14 (Channel-Attentive BiGRU)
    WORKING_DIR = "./working/idea_14"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Data Files (Parquet format from metadata generation)
    TRAIN_FILE = os.path.join(METADATA_DIR, "train.parquet")
    VAL_FILE = os.path.join(METADATA_DIR, "val.parquet")
    TEST_FILE = os.path.join(METADATA_DIR, "test.parquet")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Cache Files
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_data.pt")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_data.pt")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_data.pt")

    # Model Checkpoint
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # =========================================================================
    # Data Specifications
    # =========================================================================
    SEQ_LEN = 107
    PRED_LEN = 68

    # Targets to be predicted and scored
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    NUM_TARGETS = len(TARGET_COLS)

    # Vocabularies
    # Nucleotides: A, G, C, U (plus padding/unknown if needed, but usually 0-3)
    VOCAB_SIZE = 4
    # Loop types: S, M, I, B, H, E, X
    LOOP_VOCAB_SIZE = 7

    # =========================================================================
    # Model Architecture
    # =========================================================================
    # Embedding dimensions
    EMBED_DIM = 128

    # Main Backbone (BiGRU)
    HIDDEN_DIM = 384  # High hidden dimension as per strategy
    N_LAYERS = 6  # Deep network
    DROPOUT = 0.1

    # Squeeze-and-Excitation (SE) Block
    SE_REDUCTION = 16  # Reduction ratio for the bottleneck in SE block

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 32  # Adjusted for A100 memory and model size
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    MAX_GRAD_NORM = 1.0
    PATIENCE = 10  # Early stopping patience

    # Scheduler
    T_MAX = EPOCHS  # For CosineAnnealingLR
    ETA_MIN = 1e-6

    # =========================================================================
    # Hardware & Reproducibility
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    SEED = 2024

    @staticmethod
    def set_seed(seed=SEED):
        """
        Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
        """
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Ensure deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
