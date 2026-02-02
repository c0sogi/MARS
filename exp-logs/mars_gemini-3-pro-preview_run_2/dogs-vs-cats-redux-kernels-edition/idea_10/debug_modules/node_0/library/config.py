import os
import torch


class Config:
    """
    Configuration class for the Heterogeneous Stacking Strategy with Dual-Mix Regularization.
    """

    # =========================================================================
    # System Settings
    # =========================================================================
    SEED = 42
    NUM_WORKERS = 12  # Utilizing available vCPUs
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output Directories
    # Using 'idea_10' as the specific working directory for this experiment
    WORKING_DIR = "./working/idea_10"
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    OOF_DIR = os.path.join(WORKING_DIR, "oof")  # For Stacking Out-Of-Fold predictions
    CACHE_DIR = os.path.join(
        WORKING_DIR, "cache"
    )  # For deterministic data processing cache

    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Configuration
    # =========================================================================
    IMAGE_SIZE = 224  # Fixed resolution as per strategy
    NUM_CLASSES = 1  # Binary Classification (Dog=1, Cat=0)

    # Augmentation Constraints
    RRC_MIN_SCALE = 0.8  # RandomResizedCrop minimum scale
    RRC_MAX_SCALE = 1.0

    # Dual-Mix Regularization
    MIXUP_ALPHA = 0.4
    CUTMIX_ALPHA = 1.0
    MIXUP_PROB = 0.5
    CUTMIX_PROB = 0.5

    # =========================================================================
    # Model Configuration
    # =========================================================================
    # Base Learners for Heterogeneous Ensemble
    MODEL_NAMES = ["convnext_small.fb_in22k", "swin_small_patch4_window7_224"]

    # Meta Learner for Stacking
    META_MODEL = "LogisticRegression"

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    N_FOLDS = 5
    EPOCHS = 20

    # Optimization
    BATCH_SIZE = 64  # A100-40GB can handle this easily for 'small' models
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2  # Standard for AdamW

    # Scheduler (Cosine Annealing)
    T_MAX = 20  # Matches Epochs
    MIN_LR = 1e-6

    # Early Stopping
    # Strategy dictates disabling early stopping or setting patience to full duration
    # to allow Cosine Annealing to finish.
    PATIENCE = 20

    # =========================================================================
    # Utility Methods
    # =========================================================================
    @classmethod
    def setup_directories(cls):
        """
        Ensures all necessary directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.OOF_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
