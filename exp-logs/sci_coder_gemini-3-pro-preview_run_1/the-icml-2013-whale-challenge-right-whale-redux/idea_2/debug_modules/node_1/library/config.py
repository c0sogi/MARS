import os
import torch


class Config:
    # ==========================================
    # General Configuration
    # ==========================================
    SEED = 42
    NUM_WORKERS = 4  # Number of subprocesses for data loading
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # Data Configuration
    # ==========================================
    SR = 2000  # Sample rate (Hz)
    DURATION = 2.0  # Duration of clips (seconds)
    N_MELS = 64  # Number of Mel bands
    N_FFT = 256  # FFT window size (>= win_length)
    WIN_LENGTH = 50  # Window length: 25ms * 2000Hz = 50 samples
    HOP_LENGTH = 20  # Hop length: 10ms * 2000Hz = 20 samples
    F_MIN = 20  # Minimum frequency
    F_MAX = 1000  # Maximum frequency (Nyquist for 2kHz SR)

    # Calculated parameters
    FIXED_NUM_SAMPLES = int(SR * DURATION)  # 4000 samples

    # ==========================================
    # Augmentation Configuration
    # ==========================================
    # SpecAugment parameters
    SPEC_AUG_FREQ_MASK_PARAM = 10
    # Constraint: Max time mask width 200ms.
    # 200ms / 10ms hop = 20 frames.
    SPEC_AUG_TIME_MASK_PARAM = 20
    SPEC_AUG_NUM_FREQ_MASKS = 1
    SPEC_AUG_NUM_TIME_MASKS = 1

    # ==========================================
    # Model Configuration
    # ==========================================
    RESNET_ARCH = "resnet18"  # Backbone architecture
    GRU_HIDDEN_DIM = 128  # Hidden dimension for BiGRU
    GRU_NUM_LAYERS = 2  # Number of GRU layers
    ATTENTION_DIM = 128  # Dimension for Attention mechanism
    DROPOUT = 0.3  # Dropout rate
    NUM_CLASSES = 1  # Binary classification

    # ==========================================
    # Training Configuration
    # ==========================================
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 30

    # Class Imbalance Handling
    # Approx 1:9 ratio (Positive:Negative)
    POS_WEIGHT = 9.0

    # Optimization
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_PATIENCE = 8
    LR_SCHEDULER_PATIENCE = 3
    LR_SCHEDULER_FACTOR = 0.5

    # ==========================================
    # Paths & Directories
    # ==========================================
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Working Directory for Caching and Models
    WORKING_DIR = "./working/idea_2"

    # Cache Files (npy format)
    TRAIN_DATA_CACHE = os.path.join(WORKING_DIR, "train_data.npy")
    TRAIN_LABELS_CACHE = os.path.join(WORKING_DIR, "train_labels.npy")
    VAL_DATA_CACHE = os.path.join(WORKING_DIR, "val_data.npy")
    VAL_LABELS_CACHE = os.path.join(WORKING_DIR, "val_labels.npy")
    TEST_DATA_CACHE = os.path.join(WORKING_DIR, "test_data.npy")
    TEST_IDS_CACHE = os.path.join(WORKING_DIR, "test_ids.npy")

    # Model Checkpoint
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    @staticmethod
    def setup_dirs():
        """Ensure necessary directories exist."""
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
