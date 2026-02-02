import os
import torch


class Config:
    """
    Centralized configuration for the Lung Function Decline prediction task.
    Implements settings for the Representative Slice 2D CNN-MLP solution.
    """

    # ====================================================
    # General Settings
    # ====================================================
    SEED = 42
    DEBUG = False  # Set to True to use a small subset of data for debugging
    EXPERIMENT_NAME = "idea_1_representative_slice"

    # Compute
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Number of workers for DataLoader (adjusted for 12 vCPUs)
    NUM_WORKERS = 4

    # ====================================================
    # File Paths
    # ====================================================
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Directories
    WORKING_DIR = "./working/idea_1"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    MODEL_CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ====================================================
    # Data Preprocessing & Heuristics
    # ====================================================
    # Image Settings
    IMG_SIZE = 256

    # DICOM Windowing (Lung Window)
    WINDOW_CENTER = -600
    WINDOW_WIDTH = 1500

    # "Representative Slice" Selection Heuristic
    # We select the slice with the most pixels in this HU range (Lung Tissue)
    LUNG_MIN_HU = -1000
    LUNG_MAX_HU = -400

    # Tabular Data Engineering
    # 'Baseline_FVC' will be engineered during data loading
    NUMERICAL_FEATURES = ["Weeks", "Percent", "Age", "Baseline_FVC"]
    CATEGORICAL_FEATURES = ["Sex", "SmokingStatus"]

    # ====================================================
    # Model Architecture
    # ====================================================
    # Using EfficientNet-B0 (Noisy Student weights) via timm
    MODEL_NAME = "tf_efficientnet_b0_ns"
    PRETRAINED = True

    # Training Strategy: Freeze CNN backbone, train MLP head + fusion
    FREEZE_BACKBONE = True

    # Output: [Prediction (FVC), Uncertainty (Sigma)]
    OUTPUT_DIM = 2

    # ====================================================
    # Training Hyperparameters
    # ====================================================
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5

    # Scheduler
    T_MAX = 20  # For CosineAnnealingLR
    ETA_MIN = 1e-5

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 5

    # ====================================================
    # Metric & Loss Constants
    # ====================================================
    # Metric: Laplace Log Likelihood
    # Constants for metric calculation (clipping)
    SIGMA_CLIP = 70
    ERROR_MAX = 1000

    # Target Normalization Constants
    TARGET_MEAN = 2650.0
    TARGET_STD = 800.0

    @classmethod
    def setup(cls):
        """
        Ensures all necessary working directories exist.
        Should be called at the start of the pipeline.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print(f"--- Configuration: {cls.EXPERIMENT_NAME} ---")
        print(f"Device: {cls.DEVICE}")
        print(f"Model: {cls.MODEL_NAME} (Frozen Backbone: {cls.FREEZE_BACKBONE})")
        print(f"Image Size: {cls.IMG_SIZE}x{cls.IMG_SIZE}")
        print(
            f"Batch Size: {cls.BATCH_SIZE}, Epochs: {cls.EPOCHS}, LR: {cls.LEARNING_RATE}"
        )
        print(f"Debug Mode: {cls.DEBUG}")
        print("-" * 40)
