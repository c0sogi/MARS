import os
import torch


class Config:
    # ==========================================
    # Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for caching processed data and checkpoints
    # Explicitly using idea_48 as requested for cache invalidation/isolation
    WORK_DIR = "./working/idea_48"
    CACHE_DIR = os.path.join(WORK_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Data Processing Parameters
    # ==========================================
    WINDOW_SIZE = 64
    STRIDE = 32
    NUM_CLASSES = 21

    # Skeleton Processing
    NUM_JOINTS = 20
    # 3 (Pos) + 3 (Vel) + 3 (Acc) = 9 channels per joint
    CHANNELS_PER_JOINT = 9
    INPUT_DIM_SKELETON = NUM_JOINTS * CHANNELS_PER_JOINT  # 180

    # Audio Processing
    AUDIO_SAMPLERATE = 16000
    N_MFCC = 13
    INPUT_DIM_AUDIO = N_MFCC

    # Augmentation
    NOISE_SIGMA = 0.01  # Gaussian noise sigma for position injection

    # ==========================================
    # Model Architecture Parameters
    # ==========================================
    # Stage 1: Bi-GRU
    HIDDEN_SIZE = 96
    GRU_LAYERS = 2
    DROPOUT = 0.4

    # Stage 2 & 3: TCN Refinement
    # Monotonically increasing dilation schedule
    DILATIONS = [1, 2, 4, 8, 16]
    KERNEL_SIZE = 3
    STOCHASTIC_DEPTH_PROB = 0.2  # p_drop for DropPath

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 60
    EARLY_STOPPING_PATIENCE = 10

    # Loss Configuration
    LABEL_SMOOTHING = 0.1
    LOG_SMOOTHING_WEIGHT = 0.15
    LOG_SMOOTHING_THRESHOLD = 1.0

    # ==========================================
    # Inference & Post-Processing
    # ==========================================
    MIN_GESTURE_LENGTH = 5  # Frames

    # ==========================================
    # Debugging & Resource Control
    # ==========================================
    # Set to True to run on a small subset of data
    DEBUG = False
    DEBUG_SUBSET_SIZE = 10

    # Device configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    @classmethod
    def print_config(cls):
        print("=" * 30)
        print("Configuration:")
        print(f"  Device: {cls.DEVICE}")
        print(f"  Window Size: {cls.WINDOW_SIZE}")
        print(f"  Stride: {cls.STRIDE}")
        print(f"  Hidden Size: {cls.HIDDEN_SIZE}")
        print(f"  Dropout: {cls.DROPOUT}")
        print(f"  Stochastic Depth Prob: {cls.STOCHASTIC_DEPTH_PROB}")
        print(f"  Label Smoothing: {cls.LABEL_SMOOTHING}")
        print(f"  Noise Sigma: {cls.NOISE_SIGMA}")
        print(f"  Work Dir: {cls.WORK_DIR}")
        print("=" * 30)
