import os
import torch


class Config:
    """
    Centralized configuration for the Calibrated Full-Fit ResNet-18 Ensemble solution.
    Handles paths, hyperparameters, and system settings.
    """

    # --- Directory Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_12"

    # --- File Paths ---
    # Raw Data
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata (Pre-generated)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Artifacts
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # Phase 1 Artifacts (Trajectory & Calibration)
    # Stores the optimal epoch and scheduler milestones
    TRAJECTORY_PATH = os.path.join(WORKING_DIR, "trajectory_info.json")
    # Stores the Platt Scaling (Logistic Regression) model
    CALIBRATION_PATH = os.path.join(WORKING_DIR, "calibration_model.pkl")
    # Directory to save model checkpoints
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")

    # --- System Settings ---
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # --- Data Configuration ---
    IMG_SIZE = 224  # Upsampled from 75x75 via Bicubic
    IN_CHANNELS = 3  # Composite Band Fusion (Band1, Band2, Avg)
    BATCH_SIZE = 32  # Strictly enforced for gradient noise stability

    # --- Model Architecture ---
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    DROPOUT_RATE = 0.5  # For the minimalist head
    NUM_CLASSES = 1  # Binary classification (Ship vs Iceberg)

    # --- Optimization (AdamW) ---
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01

    # --- Scheduler (ReduceLROnPlateau / MultiStepLR) ---
    SCHEDULER_PATIENCE = 3
    SCHEDULER_FACTOR = 0.1
    MIN_LR = 1e-6

    # --- Training Protocol ---
    NUM_FOLDS = 5
    MAX_EPOCHS = 50  # Upper bound; controlled by Early Stopping in Phase 1
    EARLY_STOPPING_PATIENCE = 10

    # --- Augmentation ---
    ROTATION_LIMIT = 20  # Degrees (+/- 20)

    @classmethod
    def setup(cls):
        """Ensures necessary working directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)


# Initialize directories upon module import
Config.setup()
