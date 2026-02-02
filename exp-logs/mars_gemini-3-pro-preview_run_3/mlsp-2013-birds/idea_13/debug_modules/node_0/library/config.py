import os
import torch


class Config:
    """
    Central configuration for the Bird Species Classification Task.
    Implements settings for the Dual-Stream Heterogeneous Ensemble with Model EMA.
    """

    # ==========================================
    # General Settings
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLES = 20  # Number of samples to use when DEBUG is True

    # ==========================================
    # Paths
    # ==========================================
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"

    # Input Data Directories
    # Stream 1: Standard Spectrograms
    SPECTROGRAM_DIR = os.path.join(INPUT_ROOT, "supplemental_data", "spectrograms")
    # Stream 2: Filtered Spectrograms (Noise Suppressed)
    FILTERED_SPECTROGRAM_DIR = os.path.join(
        INPUT_ROOT, "supplemental_data", "filtered_spectrograms"
    )

    # Output Directories
    WORKING_DIR = "./working/idea_13"
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Configuration
    # ==========================================
    # Dual-Stream Sources configuration
    DATA_SOURCES = [
        {"name": "standard", "path": SPECTROGRAM_DIR},
        {"name": "filtered", "path": FILTERED_SPECTROGRAM_DIR},
    ]

    IMG_SIZE = (224, 224)
    NUM_CHANNELS = 3  # Input images will be converted to 3 channels
    NUM_CLASSES = 19
    N_FOLDS = 5

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Ensemble Components
    ARCHITECTURES = ["resnet18", "efficientnet_b0", "densenet121"]

    # Optimization
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 20

    # Regularization
    MIXUP_ALPHA = 0.4
    EMA_DECAY = 0.999  # Exponential Moving Average decay for model weights

    # ==========================================
    # Compute
    # ==========================================
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Creates necessary output directories for the experiment.
        Should be called at the start of the pipeline.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
