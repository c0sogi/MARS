import os
import torch


class Config:
    """
    Configuration class for the Stomach and Intestines MRI Segmentation task.
    Centralizes all file paths, hyperparameters, and model settings.
    """

    # =========================
    # General Settings
    # =========================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    EXPERIMENT_NAME = "idea_3"

    # =========================
    # Directories & Paths
    # =========================
    # Input directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train")
    TEST_IMG_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output directories
    WORKING_DIR = os.path.join("./working", EXPERIMENT_NAME)
    MODELS_DIR = os.path.join(WORKING_DIR, "models")
    PREDICTIONS_DIR = os.path.join(WORKING_DIR, "predictions")
    SUBMISSION_DIR = "./submission"

    # Checkpoint paths
    BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pth")
    LAST_MODEL_PATH = os.path.join(MODELS_DIR, "last_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Create directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================
    # Data Preprocessing
    # =========================
    # Image resolution (Height, Width) - divisible by 32 for U-Net
    IMG_SIZE = (320, 384)

    # 2.5D Stacking
    IN_CHANNELS = 3  # (slice_t-1, slice_t, slice_t+1)

    # Normalization (Robust Per-Slice)
    # Percentiles for clipping
    NORM_MIN_PERCENTILE = 1.0
    NORM_MAX_PERCENTILE = 99.0

    # =========================
    # Model Architecture
    # =========================
    ARCH = "UnetPlusPlus"
    BACKBONE = "efficientnet_b4"
    ENCODER_WEIGHTS = "imagenet"
    CLASSES = ["large_bowel", "small_bowel", "stomach"]
    NUM_CLASSES = len(CLASSES)
    ACTIVATION = None  # Logits are returned, activation applied in loss/metric

    # =========================
    # Training Hyperparameters
    # =========================
    BATCH_SIZE = 32  # Tuned for A100 40GB with EffNet-B4 and 320x384
    EPOCHS = 15
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-5

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS
    MIN_LR = 1e-6

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 12
    MIXED_PRECISION = True  # Use AMP

    # =========================
    # Loss & Metrics
    # =========================
    # Loss weights
    BCE_WEIGHT = 0.5
    TVERSKY_WEIGHT = 0.5

    # Tversky Loss params
    TVERSKY_ALPHA = 0.5
    TVERSKY_BETA = 0.5
    TVERSKY_SMOOTH = 1.0

    # Competition Metric Weights
    METRIC_DICE_WEIGHT = 0.4
    METRIC_HAUSDORFF_WEIGHT = 0.6

    # Post-processing
    # Threshold for converting probability to binary mask
    MASK_THRESHOLD = 0.5
    # Minimum size for removing small connected components (in pixels)
    MIN_COMPONENT_SIZE = 50

    @classmethod
    def display(cls):
        """Prints the configuration."""
        print("=" * 30)
        print(f"Configuration: {cls.EXPERIMENT_NAME}")
        print("=" * 30)
        for key, value in cls.__dict__.items():
            if not key.startswith("__") and not callable(value):
                print(f"{key}: {value}")
        print("=" * 30)
