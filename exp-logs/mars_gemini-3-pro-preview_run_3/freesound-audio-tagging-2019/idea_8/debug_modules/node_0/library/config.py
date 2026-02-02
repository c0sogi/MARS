import os
import torch


class Config:
    # ==========================================
    #             File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Specific working directory for this experiment iteration
    WORKING_DIR = "./working/idea_8"

    # Subdirectories for caching specific data types
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata Files
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # ==========================================
    #          Audio Processing
    # ==========================================
    SAMPLE_RATE = 32000
    DURATION = 30  # Seconds

    # Spectrogram Parameters
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512
    F_MIN = 0
    F_MAX = None  # Will default to SAMPLE_RATE // 2

    # Calculated Input Size (Time steps)
    # 32000 * 30 / 512 = ~1875 time steps

    # ==========================================
    #          Model Architecture
    # ==========================================
    MODEL_NAME = "efficientnet_b0"
    NUM_CLASSES = 80
    PRETRAINED = True

    # ==========================================
    #          Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 32
    EPOCHS = 25  # Sufficient for convergence with OneCycleLR

    # Optimizer & Scheduler
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    MAX_LR = 1e-3

    # Regularization
    MIXUP_ALPHA = 0.4

    # Early Stopping
    PATIENCE = 5

    # ==========================================
    #          System Settings
    # ==========================================
    NUM_WORKERS = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def setup():
        """Ensure necessary directories exist."""
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)


# Initialize directories on import
Config.setup()
