import os
import torch


class Config:
    """
    Central configuration for the Physics-Augmented Shallow GRU pipeline.
    """

    # ==========================================
    # Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_1"
    SUBMISSION_DIR = "./submission"

    # Input Files (Generated Metadata)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "model.pth")

    # Cache Files (for deterministic data processing)
    # Using .npy for numpy arrays as requested (no pickle)
    # Updated filenames to force re-processing with new features
    TRAIN_CACHE_DATA = os.path.join(WORKING_DIR, "train_data_v2.npy")
    TRAIN_CACHE_TARGET = os.path.join(WORKING_DIR, "train_targets_v2.npy")
    VAL_CACHE_DATA = os.path.join(WORKING_DIR, "val_data_v2.npy")
    VAL_CACHE_TARGET = os.path.join(WORKING_DIR, "val_targets_v2.npy")
    TEST_CACHE_DATA = os.path.join(WORKING_DIR, "test_data_v2.npy")
    STATS_CACHE = os.path.join(
        WORKING_DIR, "stats_v2.npy"
    )  # Stores mean/std for normalization

    # ==========================================
    # Data Parameters
    # ==========================================
    SEQ_LEN = 80  # Fixed length of breath sequences

    # Features to be used:
    # Raw: time_step, u_in, u_out, R, C
    # Engineered: u_in_cumsum (Volume proxy), R_flow (R * u_in), C_volume (u_in_cumsum / C)
    FEATURE_COLS = [
        "time_step",
        "u_in",
        "u_out",
        "R",
        "C",
        "u_in_cumsum",
        "R_flow",
        "C_volume",
    ]
    INPUT_DIM = len(FEATURE_COLS)

    # ==========================================
    # Model Parameters
    # ==========================================
    HIDDEN_DIM = 128
    BIDIRECTIONAL = True
    NUM_LAYERS = 1
    DROPOUT = 0.0

    # ==========================================
    # Training Parameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 256
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 7  # Early stopping patience

    # ==========================================
    # Compute & Debug
    # ==========================================
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debug mode: if True, loads a small subset of data for quick pipeline testing
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 200  # Number of breaths to use in debug mode

    @classmethod
    def setup(cls):
        """
        Ensures that necessary working directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set reproducibility
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
