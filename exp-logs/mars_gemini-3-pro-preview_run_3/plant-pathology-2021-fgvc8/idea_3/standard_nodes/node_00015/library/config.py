import os
import torch


class Config:
    # ==========================================
    # Compute & Reproducibility
    # ==========================================
    SEED = 42
    NUM_WORKERS = 12
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # File Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_3"

    # Input Metadata
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output Directories & Files
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    # Final Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Configuration
    # ==========================================
    IMAGE_SIZE = 224
    BATCH_SIZE = 64

    # Labels sorted alphabetically to ensure consistent index-to-label mapping
    LABELS = [
        "complex",
        "frog_eye_leaf_spot",
        "healthy",
        "powdery_mildew",
        "rust",
        "scab",
    ]
    NUM_CLASSES = len(LABELS)

    # ==========================================
    # Augmentation Strategy
    # ==========================================
    # RandomResizedCrop lower scale limit
    AUG_SCALE_MIN = 0.5
    # Color Jitter intensity (brightness/contrast)
    AUG_COLOR_JITTER = 0.2

    # ==========================================
    # Model Architecture
    # ==========================================
    MODEL_NAME = "convnext_tiny"
    PRETRAINED = True

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # ==========================================
    # Inference Configuration
    # ==========================================
    CONF_THRESHOLD = 0.5

    @classmethod
    def setup(cls):
        """
        Creates necessary directories for checkpoints and submissions.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
