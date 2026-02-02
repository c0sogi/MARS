import os
import torch


class Config:
    """
    Configuration for the Matched-Depth Specialist Ensemble (MDSE).
    Defines hyperparameters, paths, and structural constants for the
    three-model ensemble strategy.
    """

    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_23"
    SUBMISSION_PATH = "submission.csv"

    # Metadata File Paths
    METADATA_TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    METADATA_VAL_PATH = os.path.join(METADATA_DIR, "validation.csv")
    METADATA_TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Ensure working directory exists for caching and checkpoints
    os.makedirs(WORKING_DIR, exist_ok=True)

    # =========================================================================
    # Data Specification & Specialist Ranges
    # =========================================================================
    IMAGE_SIZE = 512
    IN_CHANNELS = 3

    # Z-Ranges for the three specialist models.
    # Each range covers 24 slices.
    # Input construction uses the "Overlapping Thick Slab" strategy:
    # 3 channels of 12 slices each, with a stride of 6.
    # Channel 1: [Start, Start + 12)
    # Channel 2: [Start + 6, Start + 18)
    # Channel 3: [Start + 12, Start + 24)
    Z_RANGES = {
        "High": (16, 40),  # Focus: Upper layers (Ink in Ch 2/3)
        "Mid": (20, 44),  # Focus: Middle layers (Ink in Ch 2)
        "Low": (24, 48),  # Focus: Lower layers (Ink in Ch 1/2)
    }

    SLAB_DEPTH = 12
    SLAB_STRIDE = 6

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    MODEL_ARCH = "SegFormer"
    BACKBONE = "mit_b2"  # ~25M params, optimal for this dataset size
    DECODER = "MLP"  # All-MLP decoder to reduce high-freq noise
    PRETRAINED_WEIGHTS = "imagenet"

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 8  # Strictly 8 to prevent underfitting
    LEARNING_RATE = 6e-5  # Conservative LR for stability
    EPOCHS = 15
    PATIENCE = 5  # Early stopping patience
    SEED = 42

    # Loss Configuration
    LOSS_TYPE = "BCE_Dice"
    DICE_BETA = 0.5  # F0.5 score optimization (Precision > Recall)

    # =========================================================================
    # Compute & Environment
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2  # Conservative for cloud environments

    # =========================================================================
    # Inference & Debugging
    # =========================================================================
    THRESHOLD = 0.5

    # Debugging flags to control dataset size
    DEBUG = False
    MAX_TRAIN_SAMPLES = (
        None  # Set to int (e.g., 100) to limit training data for debugging
    )

    @staticmethod
    def print_config():
        """Prints the current configuration."""
        print("\n" + "=" * 40)
        print("MDSE CONFIGURATION")
        print("=" * 40)
        print(f"Device        : {Config.DEVICE}")
        print(f"Model         : {Config.MODEL_ARCH} ({Config.BACKBONE})")
        print(f"Batch Size    : {Config.BATCH_SIZE}")
        print(f"Learning Rate : {Config.LEARNING_RATE}")
        print(f"Z-Ranges      : {Config.Z_RANGES}")
        print(f"Working Dir   : {Config.WORKING_DIR}")
        print("=" * 40 + "\n")
