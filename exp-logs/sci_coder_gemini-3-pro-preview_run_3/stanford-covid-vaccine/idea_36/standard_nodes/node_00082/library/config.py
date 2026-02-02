import os
import torch


class Config:
    # ==========================================
    # General Settings
    # ==========================================
    SEED = 42
    IDEA_NAME = "idea_36"
    WORKING_DIR = os.path.join("./working", IDEA_NAME)

    # ==========================================
    # Data Paths
    # ==========================================
    METADATA_DIR = "./metadata"
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")
    SAMPLE_SUBMISSION_PATH = "./input/sample_submission.csv"

    # ==========================================
    # Output Paths
    # ==========================================
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # Cache paths for deterministic data processing
    TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_cache.npy")
    VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_cache.npy")
    TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_cache.npy")

    # ==========================================
    # Data Dimensions & Processing
    # ==========================================
    SEQ_LENGTH = 107
    SEQ_SCORED = 68

    # Input Features: 4 (ACGU) + 3 (().) + 7 (Loop Types)
    NUM_NODE_FEATURES = 14

    # Targets
    NUM_TARGETS = 5
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    SCORED_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # ==========================================
    # Model Architecture (DDPN-BiGRU)
    # ==========================================
    HIDDEN_DIM = 384
    NUM_LAYERS = 4
    DROPOUT = 0.1

    # Convolutional Stem
    STEM_KERNEL_SIZE = 3
    STEM_FILTERS = 256

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 50
    PATIENCE = 10  # Early stopping patience
    GRADIENT_CLIP = 1.0  # Mandatory for stability

    # ==========================================
    # Hardware
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2

    @classmethod
    def setup(cls):
        """Ensures the working directory exists."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)


# Execute setup immediately upon import
Config.setup()
