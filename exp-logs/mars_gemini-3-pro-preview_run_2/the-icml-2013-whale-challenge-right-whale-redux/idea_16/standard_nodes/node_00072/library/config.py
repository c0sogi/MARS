import os
import torch


class Config:
    """
    Centralized configuration for the Right Whale Detection task.
    Implements settings for Idea 16: Metric-Aligned Heterogeneous Stacked Ensemble.
    """

    # =========================================================================
    # Directories and Paths
    # =========================================================================
    INPUT_ROOT = "./input"
    TRAIN_DIR = os.path.join(INPUT_ROOT, "train2")
    TEST_DIR = os.path.join(INPUT_ROOT, "test2")

    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    SAMPLE_SUBMISSION = os.path.join(INPUT_ROOT, "sampleSubmission.csv")

    # Working Directory for Idea 16
    # This is where models, logs, and cached data will be stored
    WORKING_DIR = "./working/idea_16"
    OUTPUT_DIR = WORKING_DIR
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # =========================================================================
    # Audio Processing Parameters (The "Golden Recipe")
    # =========================================================================
    SR = 2000  # Sampling Rate (2kHz)
    N_FFT = 1024  # High frequency resolution
    HOP_LENGTH = 64  # High temporal resolution
    N_MELS = 128  # Mel bands
    FMIN = 0
    FMAX = None  # Defaults to SR // 2

    # Dynamic Range Correction
    # Critical fix: Clamps noise floor to prevent silence tail skewing
    TOP_DB = 80

    # Normalization
    # We use Instance Standardization (Zero-Mean, Unit-Var) per clip
    # So we disable built-in MelSpectrogram normalization if it conflicts
    NORMALIZED_MEL = False

    # =========================================================================
    # Model Architecture Parameters
    # =========================================================================
    # Level 0 Base Learners
    # 1. EfficientNet-B0 with Noisy Student weights (JFT-300M pre-training)
    # 2. ResNet-34 with standard ImageNet weights
    MODEL_NAMES = ["tf_efficientnet_b0.ns_jft_in1k", "resnet34"]

    NUM_CLASSES = 1
    IN_CHANNELS = 1  # Adapting first layer to 1-channel input
    USE_GEM_POOLING = True  # Generalized Mean Pooling for transient signal detection

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    N_FOLDS = 5

    # Optimization
    EPOCHS = 25  # Upper bound, controlled by Early Stopping
    BATCH_SIZE = 128  # Maximize batch size for gradient stability

    # Optimizer (AdamW)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4  # Low weight decay to preserve pre-trained features

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS  # Cycle length
    MIN_LR = 1e-6

    # Early Stopping
    # Explicitly monitoring AUC (Rank-based) instead of Loss
    EARLY_STOPPING_PATIENCE = 6
    EARLY_STOPPING_METRIC = "auc"
    EARLY_STOPPING_MODE = "max"

    # =========================================================================
    # Hardware / System
    # =========================================================================
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Creates necessary directories for the experiment.
        Should be called at the start of the execution.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        print(f"Configuration initialized. Working directory: {cls.WORKING_DIR}")
