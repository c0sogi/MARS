import os
import torch


class Config:
    """
    Global configuration for the Diabetic Retinopathy Severity Prediction pipeline.
    Implements the settings for the Multi-Scale Heterogeneous Ensemble (Idea 4).
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a subset of data for debugging
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # Optimized for 12 vCPUs

    # =========================================================================
    # Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for Idea 4 (Multi-Scale Heterogeneous Ensemble)
    OUTPUT_DIR = "./working/idea_4"

    # Final submission directory
    SUBMISSION_DIR = "./submission"

    # Metadata File Paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # =========================================================================
    # Model Specifications (Multi-Scale Heterogeneous Ensemble)
    # =========================================================================
    # Maps architecture names to their specific input resolutions.
    # - tf_efficientnet_b5_ns: High-frequency feature extraction @ 512x512
    # - convnext_base: Global structure/texture extraction @ 384x384
    MODEL_SPECS = {"tf_efficientnet_b5_ns": 512, "convnext_base": 384}

    NUM_CLASSES = 1  # Regression task (predicting continuous severity score)
    USE_GEM = True  # Use Generalized Mean Pooling

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    NUM_FOLDS = 5
    BATCH_SIZE = 32  # Tuned for A100 40GB
    EPOCHS = 10  # Sufficient for convergence with pre-trained weights

    # Optimizer settings
    LR = 1e-4
    MIN_LR = 1e-6
    WEIGHT_DECAY = 1e-6
    MAX_GRAD_NORM = 10.0

    # Scheduler settings (Cosine Annealing)
    T_MAX = EPOCHS

    # Early Stopping
    PATIENCE = 3

    # =========================================================================
    # Data Processing & Augmentation
    # =========================================================================
    # Photometric
    CLAHE_PROB = 0.5

    # Geometric
    ROTATION_DEGREES = 30

    # =========================================================================
    # Inference
    # =========================================================================
    TTA_STEPS = 2  # Test Time Augmentation: Original + Horizontal Flip

    @classmethod
    def setup(cls):
        """
        Creates necessary output directories.
        """
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories upon module import
Config.setup()
