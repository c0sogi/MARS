import os
import torch


class Config:
    """
    Configuration for Right Whale Detection (Idea 9).
    Implements parameters for Homogeneous Ensemble of SWA EfficientNet-B0s.
    """

    # ==========================================
    # General Setup
    # ==========================================
    PROJECT_NAME = "RightWhaleDetection_Idea9"
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SUBSET_SIZE = 500

    # ==========================================
    # Directories & Paths
    # ==========================================
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"

    # Metadata files (pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for this specific idea (Idea 9)
    # Caching and model checkpoints will be stored here
    WORKING_DIR = "./working/idea_9"

    # Submission output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Audio & Spectrogram Parameters
    # ==========================================
    SR = 2000
    DURATION = 2.0  # Seconds

    # STFT Parameters (High Frequency Resolution as per strategy)
    N_FFT = 1024
    HOP_LENGTH = 64  # High Temporal Resolution

    # Mel Spectrogram Parameters
    N_MELS = 128
    FMIN = 0
    FMAX = None  # Defaults to SR // 2
    MEL_NORMALIZED = (
        False  # Explicitly disable area normalization to preserve Pink noise tilt
    )

    # ==========================================
    # Model Architecture
    # ==========================================
    # Backbone: EfficientNet-B0 with Noisy Student weights
    MODEL_NAME = "tf_efficientnet_b0.ns_jft_in1k"
    PRETRAINED = True

    # Input Adaptation
    IN_CHANNELS = 1  # Modified first layer for 1-channel input
    IMG_SIZE = None  # Use native resolution (approx 128x63), DO NOT resize

    # Head Configuration
    NUM_CLASSES = 1  # Binary classification
    USE_GEM_POOLING = True  # Generalized Mean Pooling for transient detection

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    N_FOLDS = 5  # Stratified K-Fold Cross-Validation
    EPOCHS = 15  # Sufficient for convergence + SWA phases
    BATCH_SIZE = 128  # Stabilize BN statistics
    PATIENCE = 4  # Early stopping patience

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4  # Low weight decay for Noisy Student backbone

    # Stochastic Weight Averaging (SWA)
    USE_SWA = True
    SWA_START_EPOCH = 5  # Start averaging after initial convergence
    SWA_LR = 1e-4

    # ==========================================
    # Hardware & Compute
    # ==========================================
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup_directories(cls):
        """
        Ensures that the working and submission directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
