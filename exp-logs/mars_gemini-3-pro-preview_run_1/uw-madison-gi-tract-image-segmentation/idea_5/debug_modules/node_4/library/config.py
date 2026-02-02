import os
import torch


class Config:
    """
    Configuration class for Idea 5: 2D U-Net++ with Weighted Deep Supervision
    and 4-Channel Input (RGB + Depth Map).
    """

    # =========================
    # Directories & Paths
    # =========================
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata Paths (already generated)
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Working Directories for this specific idea
    WORKING_DIR = "./working/idea_5"
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    PREDICTION_DIR = os.path.join(WORKING_DIR, "predictions")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Ensure working directories exist
    for d in [WORKING_DIR, CHECKPOINT_DIR, PREDICTION_DIR, SUBMISSION_DIR, CACHE_DIR]:
        os.makedirs(d, exist_ok=True)

    # =========================
    # Data Configuration
    # =========================
    # Image resolution: 320x320 balances detail and memory for U-Net++
    IMAGE_SIZE = (320, 320)

    # Classes
    CLASSES = ["large_bowel", "small_bowel", "stomach"]
    NUM_CLASSES = len(CLASSES)

    # Input Channels: 3 (RGB Replicated) + 1 (Relative Depth Map) = 4
    IN_CHANNELS = 4

    # Robust Normalization Percentiles (clipping range)
    NORM_MIN_PERCENTILE = 1.0
    NORM_MAX_PERCENTILE = 99.0

    # =========================
    # Model Configuration
    # =========================
    ARCH = "UnetPlusPlus"
    BACKBONE = "resnet34"
    ENCODER_WEIGHTS = "imagenet"

    # Deep Supervision settings
    DEEP_SUPERVISION = True
    # Weights for the decoder outputs [Final, Aux1, Aux2, Aux3]
    # Decaying weights to prioritize the final high-resolution output
    DS_WEIGHTS = [1.0, 0.5, 0.1, 0.1]

    # =========================
    # Training Configuration
    # =========================
    SEED = 42
    BATCH_SIZE = 32
    EPOCHS = 15

    # Optimization
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-2  # Standard for AdamW
    SCHEDULER_T_MAX = EPOCHS  # For Cosine Annealing

    # Hardware
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================
    # Post-Processing
    # =========================
    # Threshold for converting probability to binary mask
    MASK_THRESHOLD = 0.5
    # Minimum volume size (in pixels) to keep during 3D CCA
    MIN_COMPONENT_SIZE = 50

    # =========================
    # Debug / Flexibility
    # =========================
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 200  # Number of samples to use in debug mode

    @classmethod
    def set_debug_mode(cls, debug=True):
        """
        Adjusts configuration for debugging purposes.
        """
        cls.DEBUG = debug
        if debug:
            cls.EPOCHS = 2
            cls.BATCH_SIZE = 8
            print(
                f"[Config] Debug mode enabled: Epochs={cls.EPOCHS}, BatchSize={cls.BATCH_SIZE}, SampleSize={cls.DEBUG_SAMPLE_SIZE}"
            )
