import os
import torch


class Config:
    """
    Configuration for the Bridged Dense-Refined Hybrid Network (Idea 14/13 Fix).
    """

    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # File Paths
    # ==========================================
    # Input Metadata
    TRAIN_CSV = "./metadata/train.csv"
    VAL_CSV = "./metadata/val.csv"
    TEST_CSV = "./metadata/test.csv"

    # Working Directory (Explicitly requested idea_13)
    WORKING_DIR = "./working/idea_13/"

    # Cache Filenames
    # Unique identifiers to ensure cache invalidation for the new architecture
    TRAIN_CACHE_FILE = "train_data_bridged_v1.npz"
    VAL_CACHE_FILE = "val_data_bridged_v1.npz"
    TEST_CACHE_FILE = "test_data_bridged_v1.npz"

    # Output Submission
    SUBMISSION_PATH = "./submission/submission.csv"

    # ==========================================
    # Data Dimensions & Targets
    # ==========================================
    SEQ_LEN = 107
    SEQ_SCORED = 68

    # Input Channels: 4 (Seq) + 3 (Struct) + 7 (Loop)
    IN_CHANNELS = 14

    # Targets
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # Only these columns are used for the Loss and Metric
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Dense Backbone (Dilated TCN)
    GROWTH_RATE = 64
    DROPOUT = 0.1
    KERNEL_SIZE = 3
    DILATIONS = [1, 2, 4, 8, 16, 32]

    # Stacking Refinement Branch
    STACKING_KERNEL_SIZE = 3
    STACKING_LAYERS = 2

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    EPOCHS = 50
    NUM_WORKERS = 2

    # Device Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def get_cache_path(cls, filename):
        """
        Resolves the full path for a cache file and ensures the directory exists.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        return os.path.join(cls.WORKING_DIR, filename)

    @classmethod
    def get_model_path(cls):
        """
        Returns the path to save the best model.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        return os.path.join(cls.WORKING_DIR, "best_model.pth")
