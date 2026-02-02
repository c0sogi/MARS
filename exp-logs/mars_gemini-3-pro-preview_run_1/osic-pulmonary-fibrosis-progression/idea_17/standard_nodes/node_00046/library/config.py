import os
import torch


class Config:
    """
    Configuration class for the Full-Fidelity Concatenated Dual-Axis Network pipeline.
    Acts as a central source of truth for paths, hyperparameters, and constants.
    """

    # ==========================================
    # 1. General & Reproducibility
    # ==========================================
    SEED = 42
    PROJECT_NAME = "Full-Fidelity-Concatenated-Dual-Axis-Network"
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 50

    # ==========================================
    # 2. File System Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Caching directory for processed Tri-Slab inputs (Idea 17 specific)
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_17")

    # Metadata file paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output submission path
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 3. Data Processing & Augmentation
    # ==========================================
    IMG_SIZE = 224  # Native resolution for EfficientNet-B0
    NUM_SLABS = 3  # Tri-Slab configuration
    SLAB_OVERLAP = 0.15  # 15% overlap between slabs
    IN_CHANNELS = 3  # RGB (MIPs mapped to channels)

    # ==========================================
    # 4. Model Architecture
    # ==========================================
    BACKBONE = "tf_efficientnet_b0_ns"
    FEATURE_DIM = 1280  # Native output dim of EfficientNet-B0 (no bottleneck)
    TABULAR_HIDDEN_DIM = 1280  # Dimension to project tabular data up to

    # ==========================================
    # 5. Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 16
    EPOCHS = 8  # Short training to prevent overfitting
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2  # For AdamW
    PATIENCE = 6  # Strict early stopping

    # ==========================================
    # 6. Metric & Loss Constants
    # ==========================================
    MAX_ERROR = 1000.0  # Clipping threshold for absolute error
    MIN_CONFIDENCE = 70.0  # Clipping threshold for confidence (sigma)

    # ==========================================
    # 7. Hardware
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Number of dataloader workers

    @classmethod
    def setup(cls):
        """
        Initialize necessary directories.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Automatically setup directories when imported
Config.setup()
