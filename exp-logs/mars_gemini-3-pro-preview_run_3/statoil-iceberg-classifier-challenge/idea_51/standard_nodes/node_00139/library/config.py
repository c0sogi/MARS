import os
import torch


class Config:
    """
    Configuration class for the Dual-Polarity Downsampling CNN (DPD-CNN) project.
    Centralizes all hyperparameters, file paths, and settings.
    """

    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # File Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_51"
    SUBMISSION_DIR = "./submission"

    # Raw Data Files
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")

    # Metadata Files
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Submission Output
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Configuration
    # ==========================================
    IMAGE_SIZE = 75
    IN_CHANNELS = 3  # HH, HV, Synthetic Average

    # ==========================================
    # Model Architecture
    # ==========================================
    # 4-Stage Backbone channel widths (Early Expansion)
    CHANNEL_WIDTHS = [64, 128, 128, 128]

    # Regularization
    DROPOUT_RATE = 0.5

    # Activation
    LEAKY_RELU_SLOPE = 0.1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    N_FOLDS = 5
    BATCH_SIZE = 32
    EPOCHS = 75
    PATIENCE = 12

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4  # L2 Regularization

    # ==========================================
    # Inference
    # ==========================================
    USE_TTA = False  # Explicitly disable Test-Time Augmentation per strategy

    # ==========================================
    # Compute / Hardware
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2

    # ==========================================
    # Debugging / Development
    # ==========================================
    DEBUG = False
    MAX_DEBUG_SAMPLES = 100

    @classmethod
    def setup(cls):
        """
        Initializes the environment:
        1. Creates necessary working and submission directories.
        2. Sets random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set deterministic behavior
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @classmethod
    def set_debug_mode(cls, debug=True, max_samples=100, epochs=2):
        """
        Adjusts configuration for debugging purposes.

        Args:
            debug (bool): Enable debug mode.
            max_samples (int): Limit dataset size.
            epochs (int): Reduce training epochs for quick testing.
        """
        cls.DEBUG = debug
        cls.MAX_DEBUG_SAMPLES = max_samples
        if debug:
            cls.EPOCHS = epochs
            print(f"Debug mode enabled: Max Samples={max_samples}, Epochs={epochs}")
