import os
import torch


class Config:
    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42

    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    # Input Data
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Working Directory (for Caching and Checkpoints)
    # Using idea_50 as the specific experiment identifier
    WORKING_DIR = "./working/idea_50"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")

    # Final Submission Directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Preprocessing & Normalization
    # -------------------------------------------------------------------------
    # Image Preprocessing
    IMG_SIZE = 260  # Native resolution for EfficientNet-B2
    NUM_SLICES = 3  # Anchor slice + 2 boundary slices
    WINDOW_LEVEL = -600  # Lung window level (HU)
    WINDOW_WIDTH = 1500  # Lung window width (HU)

    # Normalization Statistics (Derived from EDA)
    # Target (FVC) Normalization
    TARGET_MEAN = 2654.6528
    TARGET_STD = 801.7017

    # Metric Constants
    METRIC_CLIP_SIGMA = 70  # ml

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    BACKBONE_NAME = "efficientnet_b2"
    PRETRAINED = True

    # Dimensions
    IMG_EMBED_DIM = 1408  # EfficientNet-B2 final layer output channels
    PROJECTION_DIM = 64  # Bottleneck projection dimension

    # Tabular Features
    # Baseline FVC, Time, Age, Sex, Smoking, Percent (optional, but often useful)
    # Based on description: Stream B takes All Clinical Scalars.
    # We will define the count based on the implementation of the dataset.
    # Typically: Baseline_FVC, Time, Age, Sex(1), Smoking(1), Percent(1) = 6
    TABULAR_INPUT_DIM = 6

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Batch Size & Epochs
    BATCH_SIZE = 32
    NUM_EPOCHS = 50  # Max epochs, controlled by Early Stopping

    # Learning Rates (Differential)
    LR_BACKBONE = 1e-4  # Slower learning for feature extractor
    LR_HEAD = 1e-3  # Faster learning for MLPs

    # Optimization
    WEIGHT_DECAY = 1e-2
    T_MAX = NUM_EPOCHS  # For Cosine Annealing

    # Early Stopping
    PATIENCE = 10

    # DataLoader
    NUM_WORKERS = 4

    @classmethod
    def setup(cls):
        """Creates necessary directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories immediately when module is imported
Config.setup()
