import os
import torch


class Config:
    """
    Central configuration for the Right Whale Detection task.
    Implements parameters for the 'Self-Distilled Multi-Objective Stacked Ensemble' strategy.
    """

    # =========================================================================
    # General Experiment Setup
    # =========================================================================
    PROJECT_NAME = "RightWhaleDetection"
    IDEA_NAME = "idea_21"
    SEED = 42

    # Debugging / Dataset Size Control
    # Set DEBUG to True to limit the dataset size for rapid prototyping
    DEBUG = False
    DEBUG_SAMPLES = 200  # Number of samples to use when DEBUG is True

    # =========================================================================
    # Directories & Paths
    # =========================================================================
    INPUT_ROOT = "./input"
    TRAIN_DIR = os.path.join(INPUT_ROOT, "train2")
    TEST_DIR = os.path.join(INPUT_ROOT, "test2")

    # Metadata paths (pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    SAMPLE_SUBMISSION = os.path.join(INPUT_ROOT, "sampleSubmission.csv")

    # Working directory for caching processed data, checkpoints, and logs
    WORKING_DIR = os.path.join("./working", IDEA_NAME)

    # Submission directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Signal Processing (The "Golden Recipe")
    # =========================================================================
    SR = 2000  # Sampling Rate (Hz)
    N_FFT = 1024  # FFT Window Size (High Frequency Resolution)
    HOP_LENGTH = 64  # Hop Length (High Time Resolution)
    N_MELS = 128  # Number of Mel Bands
    FMIN = 0  # Minimum Frequency
    FMAX = None  # Maximum Frequency (None = Nyquist)
    TOP_DB = 80  # Dynamic Range Correction (clamping silence)
    NORMALIZED_MEL = False  # Disable area normalization to preserve pink noise tilt

    # =========================================================================
    # Data Preprocessing & Augmentation
    # =========================================================================
    IN_CHANNELS = 1  # Mono input
    RESIZE = False  # Use native resolution (~128x63); do not resize to 224x224

    # Aggressive SpecAugment Parameters
    SPEC_AUG_FREQ_MASK_PARAM = 20
    SPEC_AUG_TIME_MASK_PARAM = 40
    SPEC_AUG_FREQ_MASK_NUM = 2
    SPEC_AUG_TIME_MASK_NUM = 2

    MIXUP = False  # Excluded based on prior lessons

    # =========================================================================
    # Model Architectures
    # =========================================================================
    # Heterogeneous Base Learners
    MODELS = ["tf_efficientnet_b0_ns", "resnet34"]
    POOLING = "gem"  # Generalized Mean (GeM) Pooling
    PRETRAINED = True  # Use ImageNet/NoisyStudent weights
    NUM_CLASSES = 1  # Binary Classification

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    NUM_FOLDS = 5
    EPOCHS = 15  # Training epochs per round
    BATCH_SIZE = 32

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4  # Low weight decay
    OPTIMIZER = "AdamW"
    SCHEDULER = "CosineAnnealingLR"
    MIN_LR = 1e-6

    # =========================================================================
    # Hardware & Execution
    # =========================================================================
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Ensures that the necessary working directories exist.
        This should be called at the start of execution.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Automatically setup directories when config is imported
Config.setup()
