import os
import torch


class Config:
    """
    Configuration class for the Siamese Spatially-Fused 2.5D Network (SSF-Net).
    Defines hyperparameters, file paths, and system settings.
    """

    # ==========================================
    # System & Reproducibility
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Adjust based on available vCPUs (12 available)

    # ==========================================
    # File Paths
    # ==========================================
    # Input Directories
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata Paths (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Working Directory (Cache & Models)
    # Using specific directory 'idea_39' as per requirements
    WORKING_DIR = "./working/idea_39"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Model Checkpoint Path
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Submission Path
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Hyperparameters
    # ==========================================
    IMG_SIZE = 224

    # Sampling Strategy: High-Density Uniform Sampling
    TOTAL_SLICES = 32

    # Dual-Stream Configuration
    SLICES_PER_VIEW = 16  # 16 Even + 16 Odd = 32 Total
    NUM_MODALITIES = 4  # FLAIR, T1w, T1wCE, T2w

    # Channels per stream input to the backbone
    # (16 slices * 4 modalities) = 64 channels
    IN_CHANS = SLICES_PER_VIEW * NUM_MODALITIES

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    BACKBONE = "efficientnet_b0"
    DROP_PATH_RATE = 0.2  # Stochastic Depth
    NUM_CLASSES = 1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 15
    PATIENCE = 5  # Early stopping patience

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print("\n" + "=" * 40)
        print(" CONFIGURATION")
        print("=" * 40)
        for k, v in cls.__dict__.items():
            if not k.startswith("__") and not callable(v):
                print(f"{k:<20}: {v}")
        print("=" * 40 + "\n")
