import os


class Config:
    # --- General Configuration ---
    SEED = 42
    DEBUG = False  # Set to True for fast debugging runs
    DEBUG_SAMPLE_SIZE = 1000  # Number of samples to use in debug mode

    # --- Data Configuration ---
    ORIGINAL_SIZE = 96  # Input image size from disk
    IMAGE_SIZE = 64  # Input size to the model (after center crop)
    NUM_CLASSES = 1

    # Dataset Statistics (Calculated from EDA)
    # Order: RGB
    MEAN = [0.7035, 0.5476, 0.6975]
    STD = [0.2388, 0.2821, 0.2159]

    # --- Training Hyperparameters ---
    BATCH_SIZE = 256  # Optimized for A100-40GB
    EPOCHS = 25  # Full convergence schedule

    # Optimizer & Scheduler
    LR = 2e-4
    MIN_LR = 1e-6
    WEIGHT_DECAY = 0.05

    # Regularization
    MIXUP_ALPHA = 0.2

    # --- Model & Ensemble Configuration ---
    NUM_FOLDS = 5
    # Heterogeneous Ensemble: ConvNeXt (Isotropic) + EfficientNetV2 (Hierarchical/MBConv)
    MODEL_NAMES = ["convnext_tiny", "tf_efficientnetv2_s"]

    # Inference
    TTA_VIEWS = 8  # Dihedral TTA

    # --- Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Train/Val Metadata
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Working Directory for Idea 8
    WORKING_DIR = "./working/idea_8"

    # Cache and Checkpoints
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # --- Hardware ---
    NUM_WORKERS = 12
    DEVICE = "cuda"

    @classmethod
    def setup(cls):
        """Creates necessary output directories."""
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
