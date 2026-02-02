import os
import torch


class Config:
    """
    Centralized configuration for the Cactus Classification task.
    Implements the settings for the Heterogeneous Geometric-Consistency Stacking Ensemble.
    """

    # ==========================================
    # Directory & Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for the specific idea implementation
    WORK_DIR = "./working/idea_34"
    SUBMISSION_DIR = "./submission"

    # Ensure necessary directories exist immediately
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Submission Output Path
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Data Caching Paths (using .npy for fast I/O)
    CACHE_TRAIN_IMGS = os.path.join(WORK_DIR, "cache_train_imgs.npy")
    CACHE_TRAIN_LABELS = os.path.join(WORK_DIR, "cache_train_labels.npy")
    CACHE_TEST_IMGS = os.path.join(WORK_DIR, "cache_test_imgs.npy")
    CACHE_TEST_IDS = os.path.join(WORK_DIR, "cache_test_ids.npy")

    # ==========================================
    # Data & Preprocessing
    # ==========================================
    IMG_SIZE = 32
    NUM_CLASSES = 1

    # Caching Logic
    # LOAD_CACHED_DATA: True = Try to load .npy files; False = Force re-processing
    LOAD_CACHED_DATA = True

    # Debugging
    DEBUG = False
    DEBUG_SUBSET_SIZE = 500  # Number of samples to use if DEBUG is True

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    N_FOLDS = 5

    # Training Duration
    EPOCHS = 30
    SWA_START_EPOCH = 20  # Start Stochastic Weight Averaging at this epoch

    # Optimization
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Regularization
    MIXUP_ALPHA = 0.2

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Optimized for 12 vCPUs

    # ==========================================
    # Model Architectures
    # ==========================================
    # The ensemble consists of these three distinct backbones
    MODEL_ARCHITECTURES = ["CactusRepVGG", "CactusResNet", "CactusMicroNeXt"]

    # ==========================================
    # Inference & Stacking
    # ==========================================
    # Stateless Execution: Do not resume from previous checkpoints to ensure clean runs
    LOAD_CHECKPOINTS = False

    # Test-Time Augmentation (TTA)
    TTA_VIEWS = 4  # Original, Horizontal Flip, Vertical Flip, 180 Rotation

    # Meta-Learner Artifacts
    META_LEARNER_PATH = os.path.join(WORK_DIR, "meta_learner_logreg.joblib")

    @classmethod
    def get_checkpoint_path(cls, model_name, fold):
        """Returns the path for saving/loading a specific model fold checkpoint."""
        return os.path.join(cls.WORK_DIR, f"{model_name}_fold{fold}.pth")

    @classmethod
    def get_oof_path(cls, model_name):
        """Returns the path for saving/loading Out-Of-Fold predictions."""
        return os.path.join(cls.WORK_DIR, f"oof_{model_name}.npy")
