import os
import torch


class Config:
    """
    Global configuration for the Cactus Identification Task.
    Implements settings for the Heterogeneous Trust-Aware Stacking Ensemble.
    """

    # ==========================================
    # Experiment Metadata
    # ==========================================
    SEED = 42
    IDEA_ID = "idea_28"

    # ==========================================
    # Compute Environment
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Using 4 workers is generally safe given 12 vCPUs
    NUM_WORKERS = 4

    # ==========================================
    # Data Paths
    # ==========================================
    INPUT_DIR = "./input"
    TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train")
    TEST_IMG_DIR = os.path.join(INPUT_DIR, "test")

    METADATA_DIR = "./metadata"
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # ==========================================
    # Working Directory & Caching
    # ==========================================
    # Specific directory for this iteration as requested
    WORKING_DIR = f"./working/{IDEA_ID}"

    # Subdirectories for organized output
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # Final submission file path
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    IMG_SIZE = 32
    NUM_CLASSES = 1  # Binary classification

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    NUM_FOLDS = 5
    EPOCHS = 30
    BATCH_SIZE = 128

    # Optimizer settings
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # ==========================================
    # Regularization & Advanced Training
    # ==========================================
    # Mixup alpha for regularization
    MIXUP_ALPHA = 0.2

    # Stochastic Weight Averaging (SWA)
    SWA_START_EPOCH = 20
    SWA_LR = 5e-4

    # Auxiliary Task (File Size Regression)
    # Weight lambda for the regression loss component
    AUX_LOSS_WEIGHT = 0.5

    @classmethod
    def setup_directories(cls):
        """
        Creates necessary subdirectories for cache, checkpoints, and submissions.
        Must be called at the start of the pipeline.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        print(f"Directories initialized at {cls.WORKING_DIR}")
