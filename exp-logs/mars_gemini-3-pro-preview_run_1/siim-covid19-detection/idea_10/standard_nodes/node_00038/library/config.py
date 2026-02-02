import os
import torch


class Config:
    """
    Configuration class for the Multi-Scale Aggregated ResNet18 U-Net experiment.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False
    NUM_WORKERS = 12  # Utilizing available vCPUs

    # =========================================================================
    # Directories & Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output Directory for "Idea 10"
    OUTPUT_DIR = "./working/idea_10"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Checkpoint Path
    MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
    SUBMISSION_PATH = "./submission/submission.csv"

    # =========================================================================
    # Data Configuration
    # =========================================================================
    IMG_SIZE = 512

    # Class Definitions
    # Note: Order matters and must match the one-hot encoding columns
    CLASSES = [
        "Negative for Pneumonia",
        "Typical Appearance",
        "Indeterminate Appearance",
        "Atypical Appearance",
    ]
    NUM_CLASSES = len(CLASSES)

    # Segmentation
    # Mask fill value is 0 to ensure consistency (occluded = background)
    MASK_FILL_VALUE = 0

    # =========================================================================
    # Model Architecture
    # =========================================================================
    ENCODER_NAME = "resnet18"
    ENCODER_WEIGHTS = "imagenet"

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 32
    EPOCHS = 20

    # Optimization
    # Linear Scaling Rule: Base LR approx 3e-5 per sample -> ~1e-3 for BS=32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS
    MIN_LR = 1e-6

    # Loss Weighting (1:10 ratio as requested)
    CLS_LOSS_WEIGHT = 1.0
    SEG_LOSS_WEIGHT = 10.0

    # =========================================================================
    # Hardware
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print(f"{'='*20} CONFIGURATION {'='*20}")
        for k, v in cls.__dict__.items():
            if not k.startswith("__") and not callable(v):
                print(f"{k}: {v}")
        print(f"{'='*55}")
