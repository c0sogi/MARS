import os
import torch


class Config:
    """
    Configuration for Bird Species Classification.
    Implements the settings for 'Heterogeneous Ensemble with SAM and Cyclic TTA'.
    """

    # ==========================================
    # Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Specific working directory for this idea's artifacts
    WORKING_DIR = "./working/idea_24"

    # Metadata File Paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Spectrogram Source
    # Strategy requires using "Filtered Spectrograms" (denoised)
    # These are located in supplemental_data/filtered_spectrograms
    USE_FILTERED_SPECTROGRAMS = True
    FILTERED_SPEC_DIR = os.path.join(
        INPUT_DIR, "supplemental_data", "filtered_spectrograms"
    )

    # ==========================================
    # Data Configuration
    # ==========================================
    # Rectangular resolution: 224 (Frequency) x 448 (Time)
    # Preserves temporal fidelity of rapid bird calls
    IMG_HEIGHT = 224
    IMG_WIDTH = 448

    # Input Channels: 3 for Pseudo-RGB (ImageNet weights compatibility)
    IN_CHANNELS = 3

    # Number of bird species
    NUM_CLASSES = 19

    # ==========================================
    # Model Configuration
    # ==========================================
    # Tri-Backbone Heterogeneous Ensemble
    BACKBONES = [
        "resnet18",  # Residual Bias
        "efficientnet_b0",  # Inverted Residual Bias
        "densenet121",  # Dense Bias
    ]

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42

    # Optimization
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Duration
    # Strategy targets 1000 update steps per fold
    TOTAL_STEPS = 1000
    # Max epochs as a fallback/safety, though loop should rely on steps
    MAX_EPOCHS = 100

    # Augmentation
    MIXUP_ALPHA = 0.4

    # Optimizer & Scheduler Types (Logic implemented in training script)
    OPTIMIZER_NAME = "AdamW"  # Wrapped by SAM
    SCHEDULER_NAME = "Constant"

    # ==========================================
    # Inference / TTA
    # ==========================================
    # Cyclic Time-Rolling shifts as fractions of image width
    # [Original, 25% shift, 50% shift, 75% shift]
    TTA_SHIFTS = [0.0, 0.25, 0.50, 0.75]

    # ==========================================
    # Compute
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2

    # ==========================================
    # Debugging / Development
    # ==========================================
    # Set DEBUG to True to run on a small subset for pipeline verification
    DEBUG = False
    DEBUG_SUBSET_SIZE = 20

    @classmethod
    def setup(cls):
        """
        Perform necessary setup operations like creating directories.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)


# Execute setup immediately when module is imported
Config.setup()
