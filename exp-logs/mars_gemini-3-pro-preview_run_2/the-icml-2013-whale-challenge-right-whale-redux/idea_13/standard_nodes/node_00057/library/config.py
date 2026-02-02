import os
import torch


class Config:
    """
    Configuration module for Right Whale Detection - Idea 13.
    Implements settings for a Multi-Resolution Stacked Ensemble.
    """

    # -------------------------------------------------------------------------
    # General & Reproducibility
    # -------------------------------------------------------------------------
    PROJECT_NAME = "RightWhale_MultiRes_Ensemble"
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for testing
    DEBUG_SIZE = 200  # Number of samples to use when DEBUG is True

    # Compute Environment
    NUM_WORKERS = 4  # Optimized for available vCPUs
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # Directory & File Paths
    # -------------------------------------------------------------------------
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific experiment (Idea 13)
    # Used for caching processed data and saving model checkpoints
    WORKING_DIR = "./working/idea_13"
    SUBMISSION_DIR = "./submission"

    # Ensure necessary writeable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Input Data Directories
    TRAIN_DIR = os.path.join(INPUT_ROOT, "train2")
    TEST_DIR = os.path.join(INPUT_ROOT, "test2")

    # Metadata Files (Pre-generated)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Submission Files
    SAMPLE_SUBMISSION = os.path.join(INPUT_ROOT, "sampleSubmission.csv")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Audio Processing Parameters
    # -------------------------------------------------------------------------
    SAMPLE_RATE = 2000
    DURATION = 2.0  # Target duration in seconds (2000 * 2 = 4000 samples)

    # Spectrogram Generation
    N_FFT = 1024  # High frequency resolution
    N_MELS = 128  # Number of Mel bands
    FMIN = 0
    FMAX = None  # Defaults to Nyquist (Sample Rate / 2)

    # Multi-Resolution Streams
    # Stream 1: Standard Resolution (approx 128x63)
    HOP_LENGTH_STANDARD = 64
    # Stream 2: High Temporal Resolution (approx 128x126)
    HOP_LENGTH_HIGH_RES = 32

    # -------------------------------------------------------------------------
    # Model Architecture & Ensemble Configuration
    # -------------------------------------------------------------------------
    # Level 0: Base Learners

    # Model A: EfficientNet-B0 (Noisy Student) - Standard View
    # Capitalizes on pre-training and standard spectral features.
    MODEL_A = {
        "name": "effnet_b0_ns_std",
        "arch": "tf_efficientnet_b0_ns",
        "hop_length": HOP_LENGTH_STANDARD,
        "in_channels": 1,
        "pretrained": True,
    }

    # Model B: ResNet34 - Standard View
    # Provides architectural diversity (Legacy ResNet vs MBConv).
    MODEL_B = {
        "name": "resnet34_std",
        "arch": "resnet34",
        "hop_length": HOP_LENGTH_STANDARD,
        "in_channels": 1,
        "pretrained": True,
    }

    # Model C: EfficientNet-B0 (Noisy Student) - High-Res View
    # Structural Innovation: Captures transient details via doubled temporal resolution.
    MODEL_C = {
        "name": "effnet_b0_ns_highres",
        "arch": "tf_efficientnet_b0_ns",
        "hop_length": HOP_LENGTH_HIGH_RES,
        "in_channels": 1,
        "pretrained": True,
    }

    # List of models to train for the ensemble
    BASE_MODELS = [MODEL_A, MODEL_B, MODEL_C]

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    NUM_FOLDS = 5
    BATCH_SIZE = 128
    EPOCHS = 25

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4  # Low weight decay to preserve transfer learning features

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS
    MIN_LR = 1e-6

    # Early Stopping Strategy
    # We monitor Validation LOSS (not AUC) to ensure well-calibrated probabilities
    # which are essential for the subsequent Stacking Meta-Learner.
    EARLY_STOPPING_PATIENCE = 5
    EARLY_STOPPING_MONITOR = "val_loss"
    EARLY_STOPPING_MODE = "min"

    # Level 1: Meta-Learner (Stacking)
    META_SOLVER = "lbfgs"
    META_C = 1.0  # Regularization strength for Logistic Regression
