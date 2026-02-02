import os
import torch


class Config:
    # =========================================================================
    # General Configuration
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data for debugging
    DEBUG_SAMPLES = 50  # Number of samples to use when DEBUG is True

    # =========================================================================
    # Directories & File Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")

    # Working directory for this specific experiment (Idea 13)
    WORKING_DIR = "./working/idea_13"

    # Checkpoint storage
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")

    # Metadata files (pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Final submission output
    SUBMISSION_PATH = "./working/submission.csv"

    # =========================================================================
    # Data Configuration
    # =========================================================================
    # Target columns for Multi-Label Decomposition
    # We predict 'rust' and 'scab' as independent binary targets.
    # Healthy = (0, 0), Multiple = (1, 1)
    TARGET_COLS = ["rust", "scab"]

    # Original columns for reference/reconstruction
    ORIGINAL_TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]

    N_FOLDS = 5

    # =========================================================================
    # Model Architecture (Heterogeneous Ensemble)
    # =========================================================================
    # List of backbones to train. Each will be trained on all folds.
    MODELS = [
        {
            "name": "tf_efficientnetv2_l.in21k_ft_in1k",
            "img_size": 480,
            "batch_size": 8,  # Adjusted for A100 memory (large model/res)
            "dropout_rates": [0.0, 0.1, 0.2, 0.3, 0.4],  # Multi-Sample Dropout rates
        },
        {
            "name": "convnext_base.fb_in22k_ft_in1k_384",
            "img_size": 384,
            "batch_size": 16,
            "dropout_rates": [0.0, 0.1, 0.2, 0.3, 0.4],  # Multi-Sample Dropout rates
        },
    ]

    # Generalized Mean Pooling parameter (initial value)
    GEM_P = 3.0

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    EPOCHS = 40
    LEARNING_RATE = 1e-4
    MIN_LR = 1e-6
    WEIGHT_DECAY = 1e-4
    LABEL_SMOOTHING = 0.05

    # Stochastic Weight Averaging (SWA)
    USE_SWA = True
    SWA_START_EPOCH = 30  # Start SWA in the last 25% of training
    SWA_LR = 2e-5

    # Early Stopping
    PATIENCE = 10

    # =========================================================================
    # Augmentation (Albumentations)
    # =========================================================================
    # CoarseDropout settings to force distributed feature learning
    AUG_COARSE_DROPOUT_MAX_HOLES = 8
    AUG_COARSE_DROPOUT_MAX_HEIGHT = 100
    AUG_COARSE_DROPOUT_MAX_WIDTH = 100
    AUG_COARSE_DROPOUT_MIN_HOLES = 1
    AUG_COARSE_DROPOUT_MIN_HEIGHT = 16
    AUG_COARSE_DROPOUT_MIN_WIDTH = 16

    # =========================================================================
    # Meta-Learner (Stacking)
    # =========================================================================
    # Parameters for the Rank-Calibrated Stacking
    META_MODEL = "LogisticRegression"

    # =========================================================================
    # Hardware
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 8  # Optimized for 12 vCPUs

    @classmethod
    def create_dirs(cls):
        """
        Creates the necessary working directories for the experiment.
        Should be called at the start of the pipeline.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
