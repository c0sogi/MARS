import os
import torch


class Config:
    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data for debugging
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Number of dataloader workers

    # ==========================================
    # File Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific idea/experiment
    WORK_DIR = "./working/idea_29"
    CACHE_DIR = os.path.join(WORK_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"

    # Metadata files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output paths
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Data Processing Hyperparameters
    # ==========================================
    # Video/Skeleton
    FPS = 20
    SKELETON_JOINTS = 20
    SKELETON_CHANNELS = 3  # X, Y, Z

    # Audio (Physics-Based)
    SAMPLE_RATE = 16000
    # Hop length aligned to video frame rate: 16000 / 20 = 800 samples per frame
    AUDIO_HOP_LENGTH = 800
    AUDIO_N_FFT = 2048  # ~2.5x Hop Length
    AUDIO_N_MFCC = 13  # Compact representation

    # Normalization
    # These will be computed or loaded from stats, but we define the structure here
    USE_GLOBAL_STATS = True

    # Augmentation
    TEMPORAL_RESAMPLE_MIN = 0.8
    TEMPORAL_RESAMPLE_MAX = 1.2
    CHANNEL_MASK_PROB = 0.1

    # ==========================================
    # Model Architecture
    # ==========================================
    # Classes: 0 = Background, 1-20 = Gestures
    NUM_CLASSES = 21

    # Stems
    STEM_CHANNELS = 256
    STEM_KERNEL_SIZE = 7

    # Backbone (BiGRU)
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    DROPOUT = 0.3  # Vertical dropout rate

    # Heads
    USE_BOUNDARY_HEAD = True

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 8
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.05  # Aggressive regularization

    # Scheduler
    COSINE_T_MAX = NUM_EPOCHS

    # Loss Configuration
    LABEL_SMOOTHING = 0.1
    BACKGROUND_CLASS_WEIGHT = 0.5
    BOUNDARY_LOSS_WEIGHT = 0.5

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 10

    # Inference
    MEDIAN_FILTER_SIZE = 5
    MIN_GESTURE_LENGTH = 5

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print("-" * 30)
        print("Configuration:")
        for key, value in cls.__dict__.items():
            if not key.startswith("__") and not callable(value):
                print(f"{key}: {value}")
        print("-" * 30)
