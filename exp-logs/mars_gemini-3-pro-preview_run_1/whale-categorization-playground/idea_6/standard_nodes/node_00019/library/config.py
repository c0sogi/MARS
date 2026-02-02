import os
import torch


class Config:
    """
    Configuration class for the Whale Species Prediction task.
    Centralizes all hyperparameters, file paths, and model settings.
    """

    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_6"
    SUBMISSION_DIR = "./submission"

    # Metadata files generated in the previous step
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # =========================================================================
    # Data & Preprocessing
    # =========================================================================
    IMAGE_SIZE = 320
    NUM_CLASSES = 4029  # Total unique classes (including new_whale) from analysis
    NUM_WORKERS = 4
    PIN_MEMORY = True

    # Augmentation Hyperparameters (Conservative Strategy)
    # Aggressive distortions are avoided to preserve feature integrity
    AUG_ROTATION = 20  # degrees (+/-)
    AUG_SCALE_MIN = 0.9
    AUG_SCALE_MAX = 1.1
    AUG_BRIGHTNESS = 0.2
    AUG_CONTRAST = 0.2
    # Note: Hue and Saturation are explicitly excluded per strategy

    # =========================================================================
    # Model Architecture
    # =========================================================================
    BACKBONE = "densenet169"
    EMBEDDING_SIZE = 512
    PROJECTION_DROPOUT = 0.3
    PRETRAINED = True

    # =========================================================================
    # ArcFace Head
    # =========================================================================
    ARCFACE_MARGIN = 0.50
    ARCFACE_SCALE = 30.0

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    BATCH_SIZE = 64  # A100 40GB allows for larger batch size
    EPOCHS = 25
    LEARNING_RATE = 3e-4  # Initial LR for AdamW
    WEIGHT_DECAY = 1e-4
    MIN_LR = 1e-6  # For Cosine Annealing scheduler
    LABEL_SMOOTHING = 0.0  # Explicitly disabled for ArcFace

    # =========================================================================
    # Compute & Optimization
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_AMP = True  # Automatic Mixed Precision for faster training

    # =========================================================================
    # Debugging & Development
    # =========================================================================
    DEBUG = False  # Set to True to run on a small subset for verification
    DEBUG_SUBSET_SIZE = 500

    @classmethod
    def setup(cls):
        """
        Create necessary working directories.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    def __init__(self, **kwargs):
        """
        Allow updating configuration via init arguments for flexibility.
        """
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
