import os
import torch


class Config:
    # ==========================================
    # Directories & Paths
    # ==========================================
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train2")
    TEST_DIR = os.path.join(INPUT_DIR, "test2")

    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for caching intermediate data and model checkpoints
    # Using 'idea_17' as the current experiment identifier
    WORKING_DIR = "./working/idea_17"

    # Submission directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Audio Parameters (The Golden Recipe)
    # ==========================================
    SAMPLE_RATE = 2000  # Native sample rate of the dataset
    DURATION = 2.0  # Clip duration in seconds

    # Spectrogram generation parameters
    N_FFT = 1024
    HOP_LENGTH = 64  # High overlap for temporal resolution
    N_MELS = 128  # Number of Mel bands
    FMIN = 0
    FMAX = None  # Defaults to Nyquist (1000 Hz)

    # Signal Processing
    TOP_DB = 80.0  # Dynamic range clamping to fix noise floor

    # ==========================================
    # Model Architecture
    # ==========================================
    # Triple-Architecture Heterogeneous Stacked Ensemble
    MODEL_NAMES = [
        "tf_efficientnet_b0.ns_jft_in1k",  # Noisy Student weights
        "resnet34",  # Residual Summation
        "densenet121",  # Dense Concatenation
    ]
    NUM_CLASSES = 1  # Binary classification

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    NUM_FOLDS = 5

    # Optimization
    BATCH_SIZE = 128  # Maximize for stability
    EPOCHS = 20
    LR = 1e-3
    WEIGHT_DECAY = 1e-4  # Low weight decay for NS weights

    # Convergence
    PATIENCE = 5  # Strict early stopping based on Val AUC

    # ==========================================
    # Augmentation
    # ==========================================
    FREQ_MASK_PARAM = 20  # Aggressive frequency masking

    # ==========================================
    # Compute
    # ==========================================
    NUM_WORKERS = 2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """Creates necessary directories for outputs and cache."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories on import
Config.setup()
