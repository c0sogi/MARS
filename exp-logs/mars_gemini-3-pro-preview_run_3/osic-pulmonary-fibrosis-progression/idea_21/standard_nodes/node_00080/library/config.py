import os
import torch


class Config:
    """
    Configuration for the Robust Clinical-Residual Fusion Network (RCRF-Net).
    Defines paths, hyperparameters, and constants based on the provided idea and EDA.
    """

    # -------------------------------------------------------------------------
    # General Configuration
    # -------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging

    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata files (Pre-generated)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Working Directories
    WORKING_DIR = "./working/idea_21"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Model Checkpoint
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    # -------------------------------------------------------------------------
    # Data Preprocessing & Normalization
    # -------------------------------------------------------------------------
    # Image parameters
    IMG_SIZE = 260  # EfficientNet-B2 native resolution (260x260)
    NUM_SLICES = 3  # Anchor slice + 2 boundary slices

    # Clinical feature scaling
    TIME_SCALE = 0.01  # Scale relative time (Weeks) by 0.01

    # Normalization Statistics (Derived from EDA)
    # Used for Z-scoring inputs (Age) and Targets (FVC)
    # FVC Mean: 2654.65, Std: 801.70
    # Age Mean: 67.58, Std: 6.63
    TARGET_MEAN = 2654.6528
    TARGET_STD = 801.7017
    AGE_MEAN = 67.5825
    AGE_STD = 6.6259

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    EFFNET_ARCH = "tf_efficientnet_b2"

    # Dimensions
    # Clinical Inputs: Baseline_FVC(1) + Time(1) + Age(1) + Sex(1) + Smoking(3) = 7
    # Note: Sex is binary encoded, Smoking is One-Hot (3 classes)
    CLINICAL_INPUT_DIM = 7
    CLINICAL_HIDDEN_DIM = 128
    VISUAL_HIDDEN_DIM = 128
    LATENT_DIM = 64
    DROPOUT = 0.2

    # Fine-tuning
    BACKBONE_TRAINABLE_LAYERS = 2  # Unfreeze top 2 stages

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 32
    EPOCHS = 50
    NUM_WORKERS = 4

    # Optimization
    LR_BACKBONE = 1e-4
    LR_HEADS = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler (Cosine Annealing)
    T_MAX = 50
    ETA_MIN = 1e-6

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 10

    # -------------------------------------------------------------------------
    # Compute
    # -------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """Creates necessary working directories."""
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
