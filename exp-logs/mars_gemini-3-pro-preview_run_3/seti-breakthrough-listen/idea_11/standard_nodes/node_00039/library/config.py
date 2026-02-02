import os
import torch


class Config:
    # --- Reproducibility ---
    SEED = 42

    # --- System / Hardware ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 12  # Utilizing available vCPUs

    # --- File Paths ---
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"

    # Metadata files (Pre-generated)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output Directories
    WORKING_DIR = "./working/idea_11"
    SUBMISSION_DIR = "./submission"

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Create directories if they don't exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # --- Data Configuration ---
    # Native shape is (6, 273, 256).
    # We pad height to 288 (nearest multiple of 32) for EfficientNet compatibility.
    # Width 256 is kept as is.
    # Format: (Height, Width)
    INPUT_SIZE = (288, 256)

    # Debugging flag to limit dataset size for rapid testing
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 500

    # --- Model Architecture ---
    BACKBONE_NAME = "efficientnet_b0"
    NUM_CLASSES = 1  # Binary classification (Needle vs Haystack)

    # --- Training Hyperparameters ---
    EPOCHS = 12
    BATCH_SIZE = 128

    # Optimizer (AdamW)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler (CosineAnnealingLR)
    T_MAX = 15

    # Regularization
    MIXUP_ALPHA = 0.2

    # Early Stopping
    PATIENCE = 5

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print("\n" + "=" * 40)
        print(f"{'CONFIGURATION':^40}")
        print("=" * 40)
        for key, val in cls.__dict__.items():
            if not key.startswith("__") and not callable(val):
                print(f"{key:<25}: {val}")
        print("=" * 40 + "\n")
