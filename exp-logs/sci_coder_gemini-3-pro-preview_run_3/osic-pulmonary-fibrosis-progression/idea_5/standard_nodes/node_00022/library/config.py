import os
import torch


class Config:
    """
    Central configuration for the Pulmonary Fibrosis Progression prediction task.
    Implements the Multi-Scale Wide-and-Deep 2.5D Network strategy.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    PROJECT_NAME = "idea_5"
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 20

    # =========================================================================
    # Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Files
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Working Directories
    WORKING_DIR = os.path.join("./working", PROJECT_NAME)
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Hyperparameters
    # =========================================================================
    # Image Processing
    IMG_SIZE = 224  # Input size for EfficientNet-B0
    NUM_SLICES = 3  # 2.5D approach: Apical, Middle, Basal
    SLICE_THRESHOLD = (
        0.5  # Threshold for finding Apical/Basal slices relative to max area
    )

    # Normalization Stats (Derived from EDA)
    # Used for Z-score standardization of the target and baseline FVC
    FVC_MEAN = 2654.6528
    FVC_STD = 801.7017

    # DataLoader
    BATCH_SIZE = 32
    NUM_WORKERS = 4

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    MODEL_NAME = "efficientnet_b0"
    PRETRAINED = True

    # Feature Extraction
    # We extract features from an intermediate layer (texture) and the final layer (semantic)
    PROJECTION_DIM = 64  # Dimension to project image features to

    # Tabular Features
    # 'Weeks' and 'Baseline_FVC' go to the Wide branch
    # 'Age', 'Sex', 'SmokingStatus' go to the Deep branch
    CAT_FEATURES = ["Sex", "SmokingStatus"]
    NUM_FEATURES = ["Age"]

    # Embedding dims for categorical features
    EMBED_DIMS = {
        "Sex": 2,  # Male, Female
        "SmokingStatus": 3,  # Ex-smoker, Never smoked, Currently smokes
    }

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    EPOCHS = 35
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler (Cosine Annealing)
    T_MAX = 35  # Matches EPOCHS
    ETA_MIN = 1e-6

    # Optimizer
    OPTIMIZER = "AdamW"

    # Metric / Loss
    # Modified Laplace Log Likelihood constants
    MIN_CONFIDENCE = 70.0
    MAX_ERROR = 1000.0

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """Ensures necessary directories exist."""
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories on import
Config.setup()
