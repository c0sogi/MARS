import os
import torch


class Config:
    """
    Configuration for Idea 20: Scale-Partitioned Dense Hybrid Network.
    Defines paths, hyperparameters, and constants.
    """

    # ==========================================
    # Directories & Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Specific directory for this idea's outputs and cache
    IDEA_NAME = "idea_20"
    IDEA_DIR = os.path.join(WORKING_DIR, IDEA_NAME)

    # Metadata File Paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Submission Output
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # ==========================================
    # Data Caching
    # ==========================================
    # Unique version string to force fresh data generation for this idea
    # This ensures Partner Identity and other specific features are generated correctly
    CACHE_VERSION = "partitioned_dense_v1"

    # Cache File Paths
    TRAIN_DATA_PATH = os.path.join(IDEA_DIR, f"train_data_{CACHE_VERSION}.npz")
    VAL_DATA_PATH = os.path.join(IDEA_DIR, f"val_data_{CACHE_VERSION}.npz")
    TEST_DATA_PATH = os.path.join(IDEA_DIR, f"test_data_{CACHE_VERSION}.npz")

    # ==========================================
    # Dataset Specifications
    # ==========================================
    SEQ_LENGTH = 107
    SEQ_SCORED = 68

    # All available target columns in the dataset
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # The specific columns used for scoring and loss calculation (Masked Optimization)
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Backbone settings
    HIDDEN_DIM = 64  # "Growth Rate / Channel Width to 64"
    DROPOUT = 0.1  # "Dropout (0.1) within every block"
    KERNEL_SIZE = 3

    # Dilated TCN Configuration
    # Exponentially increasing dilation rates to ensure global receptive field
    DILATION_RATES = [1, 2, 4, 8, 16, 32]

    # Scale-Partitioned Latent Gather Logic
    # We split the backbone output into Local (d=1,2,4) and Global (d=8,16,32)
    PARTITION_SPLIT_INDEX = 3  # Index in DILATION_RATES where Global starts

    # Compression settings for the partitioned gather
    COMPRESSION_CHANNELS = 32  # Dimensions for Z_local and Z_global

    # ==========================================
    # Training Settings
    # ==========================================
    SEED = 42
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    EPOCHS = 50  # Maximum epochs
    PATIENCE = 10  # Early stopping patience

    # Hardware
    NUM_WORKERS = 2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Ensures the working directory for this idea exists.
        Should be called at the start of execution.
        """
        os.makedirs(cls.IDEA_DIR, exist_ok=True)
