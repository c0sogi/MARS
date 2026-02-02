import os
import torch


class Config:
    # ==========================================
    # 1. Environment & Reproducibility
    # ==========================================
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # Optimized for the 12 vCPUs available

    # ==========================================
    # 2. File Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_4"
    SUBMISSION_DIR = "./submission"

    # Metadata Files
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_meta.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_meta.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_meta.csv")

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Paths (for deterministic processing)
    CACHE_DIR = WORKING_DIR

    # ==========================================
    # 3. Model Hyperparameters
    # ==========================================
    # Backbones for Stage 1 (Fine-tuning) & Stage 2 (Extraction)
    BACKBONE_SWIN = "swin_large_patch4_window7_224"
    BACKBONE_EFFNET = "tf_efficientnetv2_l"

    # Image Configuration
    IMG_SIZE = 224

    # Training Configuration (Stage 1: Fine-tuning)
    BATCH_SIZE = 32  # Fits A100 40GB with Swin Large
    EPOCHS = 10  # Low epoch count for fine-tuning
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 1e-6
    MAX_GRAD_NORM = 10.0

    # Stacking Configuration (Stage 2)
    N_FOLDS = 5

    # ==========================================
    # 4. Debugging & Development
    # ==========================================
    # Set DEBUG to True to run on a small subset of data
    DEBUG = False
    DEBUG_SUBSET_SIZE = 100

    @classmethod
    def setup(cls):
        """
        Ensures necessary directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        print(f"Directories ensured: {cls.WORKING_DIR}, {cls.SUBMISSION_DIR}")


# Automatically setup directories on import
Config.setup()
