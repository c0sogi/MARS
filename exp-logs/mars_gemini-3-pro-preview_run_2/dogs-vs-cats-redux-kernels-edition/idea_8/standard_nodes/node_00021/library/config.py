import os
import torch


class Config:
    """
    Global configuration for the Heterogeneous High-Capacity Ensemble (Idea 8).
    Defines hyperparameters for data processing, model architecture, and training.
    """

    # ==========================================
    # Reproducibility & Debugging
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for rapid testing
    DEBUG_SAMPLE_SIZE = 500  # Number of samples to use when DEBUG is True

    # ==========================================
    # Compute Environment
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # 12 vCPUs available; 8 is a safe balance for data loading without overhead
    NUM_WORKERS = 8

    # ==========================================
    # File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_8"
    SUBMISSION_DIR = "./submission"

    # Checkpoint directory specific to this experiment
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")

    # ==========================================
    # Data Configuration
    # ==========================================
    IMAGE_SIZE = 224
    BATCH_SIZE = 64  # Optimized for A100 40GB VRAM with Small models
    N_FOLDS = 5

    # ==========================================
    # Augmentation Strategy
    # ==========================================
    # Mixup alpha for probability calibration
    MIXUP_ALPHA = 0.2
    # Minimum scale for RandomResizedCrop to preserve semantic integrity
    CROP_SCALE_MIN = 0.8

    # ==========================================
    # Model Architectures
    # ==========================================
    # Heterogeneous ensemble combining CNN and Transformer inductive biases
    MODEL_ARCHS = [
        "convnext_small.fb_in22k",  # CNN: Strong local feature extraction
        "swin_small_patch4_window7_224",  # Transformer: Strong global context
    ]

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # Patience set to equal EPOCHS to effectively disable early stopping
    # and allow the scheduler to complete its full cycle, as per strategy.
    PATIENCE = 20

    @classmethod
    def setup(cls):
        """
        Creates the necessary working and submission directories.
        Should be called or executed upon module import.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Automatically setup directories when config is imported
Config.setup()
