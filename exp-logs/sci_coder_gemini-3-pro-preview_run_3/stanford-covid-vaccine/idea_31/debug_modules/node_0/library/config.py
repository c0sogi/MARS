import os
import torch


class Config:
    """
    Configuration class for the Stabilized Deep Channel-Gated BiGRU (SDCG-BiGRU) strategy.
    Centralizes hyperparameters, file paths, and training settings.
    """

    # ==========================================
    # General Settings
    # ==========================================
    SEED = 2024
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SUBSET_SIZE = 100
    NUM_WORKERS = 4

    # ==========================================
    # File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_31"

    # Data Paths (Parquet Metadata)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")
    CACHE_DIR = WORKING_DIR  # Directory to store cached numpy arrays

    # ==========================================
    # Data Specifications
    # ==========================================
    SEQ_LEN = 107
    PRED_LEN = 68

    # Input Features:
    # Sequence (4: A,G,C,U) + Structure (3: (,.,)) + LoopType (7: S,M,I,B,H,E,X)
    INPUT_CHANNELS = 14

    # Target Columns
    # All 5 provided in training data
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Scored Columns
    # Only these 3 are used for the competition metric (MCRMSE)
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # ==========================================
    # Model Architecture (SDCG-BiGRU)
    # ==========================================
    # Convolutional Stem
    STEM_KERNEL_SIZE = 3
    STEM_CHANNELS = 256

    # Deep Backbone
    HIDDEN_DIM = 384
    NUM_LAYERS = 4
    DROPOUT = 0.1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 50
    PATIENCE = 7  # Early stopping patience

    # Stability Mechanisms
    GRAD_CLIP_NORM = 1.0  # Mandatory gradient clipping

    # Scheduler (Cosine Annealing)
    T_MAX = 50  # Should match EPOCHS
    ETA_MIN = 1e-6

    # ==========================================
    # Hardware
    # ==========================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def setup_directories(cls):
        """
        Ensures necessary working directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)


# Execute setup on module import
Config.setup_directories()
