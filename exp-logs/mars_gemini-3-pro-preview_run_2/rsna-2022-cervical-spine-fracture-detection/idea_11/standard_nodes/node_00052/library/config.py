import os
import torch


class Config:
    # =========================================================================
    # General Settings
    # =========================================================================
    PROJECT_NAME = "idea_11"
    DEBUG = False
    SEED = 42

    # =========================================================================
    # Directories & Paths
    # =========================================================================
    # Root directories
    DATA_ROOT = "./input"
    WORKING_DIR = f"./working/{PROJECT_NAME}"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Image directories
    TRAIN_IMAGES_DIR = os.path.join(DATA_ROOT, "train_images")
    TEST_IMAGES_DIR = os.path.join(DATA_ROOT, "test_images")
    SEGMENTATION_DIR = os.path.join(DATA_ROOT, "segmentations")

    # Metadata paths (Pre-generated)
    TRAIN_METADATA = "./metadata/train_metadata.csv"
    VAL_METADATA = "./metadata/val_metadata.csv"
    TEST_METADATA = "./metadata/test_metadata.csv"

    # Other input files
    SAMPLE_SUBMISSION = os.path.join(DATA_ROOT, "sample_submission.csv")

    # Output paths
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")
    CACHE_DIR = WORKING_DIR  # For caching processed datasets

    # =========================================================================
    # Data Hyperparameters
    # =========================================================================
    # 2.5D Stacking: z-1, z, z+1
    IN_CHANNELS = 3

    # High-Density Sampling: 96 slices per study
    SEQ_LEN = 96

    # Image Resolution (EfficientNet-B4 native is usually 380, we use 384 for convenience)
    IMAGE_SIZE = (384, 384)

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    # Backbone: EfficientNet-B4 (Noisy Student weights)
    BACKBONE = "tf_efficientnet_b4.ns_jft_in1k"

    # LSTM Settings for Sequence Modeling
    LSTM_HIDDEN_SIZE = 256
    LSTM_LAYERS = 2
    LSTM_DROPOUT = 0.2

    # Head Settings
    NUM_CLASSES = 8  # C1-C7 + Patient Overall
    DROPOUT_RATE = 0.3

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    EPOCHS = 10

    # Batch size is small due to large SEQ_LEN and Backbone memory usage
    BATCH_SIZE = 2

    # Gradient Accumulation to simulate larger batch size (Effective Batch = 2 * 8 = 16)
    ACCUMULATION_STEPS = 8

    # Optimizer
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-6
    MAX_GRAD_NORM = 1.0

    # Scheduler
    T_MAX = 10  # For CosineAnnealingLR
    MIN_LR = 1e-6

    # Loss Calibration
    # Strictly 1.0 to avoid destroying probability calibration
    POS_WEIGHT = 1.0

    # =========================================================================
    # Hardware & Performance
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 12  # Utilizing all available vCPUs
    PIN_MEMORY = True

    # =========================================================================
    # Debugging / Overrides
    # =========================================================================
    @classmethod
    def setup(cls, debug=False, epochs=None):
        """
        Allows dynamic reconfiguration for debugging or specific run modes.
        """
        if debug:
            cls.DEBUG = True
            cls.EPOCHS = 2
            cls.SEQ_LEN = 32  # Reduce sequence length for faster debug cycle
            cls.IMAGE_SIZE = (256, 256)  # Reduce resolution

        if epochs is not None:
            cls.EPOCHS = epochs
