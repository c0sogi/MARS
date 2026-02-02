import os
import torch


class Config:
    """
    Global configuration for the Bird Species Classification Task (Idea 11).
    Implements the Heterogeneous Ensemble strategy with Domain-Curated Inputs.
    """

    # -------------------------------------------------------------------------
    # Paths & Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Path to the provided BMP spectrograms (supplemental data)
    # These are used instead of generating spectrograms from raw WAVs
    SPECTROGRAM_DIR = os.path.join(INPUT_DIR, "supplemental_data", "spectrograms")

    # Working directory for this specific experiment
    WORK_DIR = "./working/idea_11"

    # Sub-directories for artifacts
    CACHE_DIR = os.path.join(WORK_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")
    # Submission directory (as per task requirement to save to ./submission)
    SUBMISSION_DIR = "./submission"

    # -------------------------------------------------------------------------
    # Data Configuration
    # -------------------------------------------------------------------------
    SEED = 42
    NUM_CLASSES = 19

    # Image parameters
    # Resizing to 224x224 to match ImageNet pretraining
    IMG_SIZE = (224, 224)
    CHANNELS = 3  # We replicate the single-channel spectrogram to 3 channels

    # -------------------------------------------------------------------------
    # Training Configuration
    # -------------------------------------------------------------------------
    N_FOLDS = 5
    BATCH_SIZE = 16  # Adjusted for small dataset stability
    EPOCHS = 50  # Max epochs, controlled by early stopping

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 10  # Early stopping patience

    # Scheduler
    T_MAX = EPOCHS  # For CosineAnnealingLR

    # Debugging / Quick Run
    # If True, limits dataset size and epochs for rapid pipeline verification
    DEBUG = False
    DEBUG_SUBSET_SIZE = 32
    DEBUG_EPOCHS = 2

    # -------------------------------------------------------------------------
    # Model Configuration
    # -------------------------------------------------------------------------
    # Heterogeneous Ensemble Architectures
    # Selected for diversity of errors and feature representation
    # Simplified to ResNet18 based on Lesson 00014 and 00003 (Robust Baseline)
    ARCHITECTURES = ["resnet18"]
    PRETRAINED = True

    # -------------------------------------------------------------------------
    # Hardware
    # -------------------------------------------------------------------------
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Ensures all necessary output directories exist.
        """
        os.makedirs(cls.WORK_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories on import
Config.setup()
