import os
import torch


class Config:
    """
    Configuration for the High-Capacity Stabilized Decoupled Bias-Refined BiGRU strategy.
    Centralizes all hyperparameters, file paths, and constants.
    """

    # ==========================================
    # File Paths and Directories
    # ==========================================
    METADATA_DIR = "./metadata"
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

    INPUT_DIR = "./input"
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Working directory for idea_61
    WORKING_DIR = "./working/idea_61"
    CACHE_DIR = WORKING_DIR  # Directory to store cached processed data
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_SAVE_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # ==========================================
    # Data Specifications
    # ==========================================
    SEQ_LEN = 107
    PRED_LEN = 68  # Number of positions scored

    # Input Features:
    # 4 (Nucleotides: A, G, C, U)
    # + 3 (Structure: (, ), .)
    # + 7 (Loop Type: S, M, I, B, H, E, X)
    # = 14 Total Channels
    INPUT_DIM = 14

    # Targets
    # We train on all 5 to utilize auxiliary signal (Multi-Task Learning)
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # We validate/score mainly on these 3
    SCORED_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    OUTPUT_DIM = 5

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # High-Capacity Backbone settings
    HIDDEN_DIM = 384  # Dimension per direction (Total BiGRU hidden size = 768)
    NUM_LAYERS = 4  # Deep 4-layer backbone
    DROPOUT = 0.1

    # Convolutional Stem settings
    CONV_KERNEL_SIZE = 3
    CONV_FILTERS = 256

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 32  # Adjusted for A100 GPU memory and model size
    LEARNING_RATE = 1e-3  # AdamW default starting rate
    WEIGHT_DECAY = 1e-4
    EPOCHS = 50
    PATIENCE = 10  # Early stopping patience
    MAX_GRAD_NORM = 1.0  # Critical for stabilizing the hybrid architecture

    # ==========================================
    # Hardware & Execution
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # For DataLoader

    @classmethod
    def setup(cls):
        """
        Ensures necessary directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
