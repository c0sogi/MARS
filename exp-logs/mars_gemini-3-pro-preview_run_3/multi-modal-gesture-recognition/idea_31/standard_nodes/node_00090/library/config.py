import os

# Ensure necessary directories exist immediately upon import
_WORKING_DIR = "./working"
_CACHE_DIR = os.path.join(_WORKING_DIR, "idea_31")
_SUBMISSION_DIR = "./submission"

os.makedirs(_CACHE_DIR, exist_ok=True)
os.makedirs(_SUBMISSION_DIR, exist_ok=True)


class Config:
    """
    Configuration for the Hierarchically-Normalized Gated-Kinematic Network (HNG-KN).
    Acts as the single source of truth for hyperparameters and paths.
    """

    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data
    DEBUG_SAMPLES = 20  # Number of samples to use in debug mode
    NUM_WORKERS = 4  # Number of dataloader workers

    # ==========================================
    # Directory Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = _WORKING_DIR
    CACHE_DIR = _CACHE_DIR
    SUBMISSION_DIR = _SUBMISSION_DIR

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Files
    MODEL_SAVE_PATH = os.path.join(CACHE_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Processing Parameters
    # ==========================================
    # Temporal Windowing
    WINDOW_SIZE = 64
    STRIDE = 32

    # Audio Features
    AUDIO_SR = 16000
    N_MFCC = 13
    N_FFT = 2048
    HOP_LENGTH = 512

    # Skeleton Features
    # 20 Joints * 3 Coordinates (X,Y,Z) * 3 Derivatives (Pos, Vel, Acc)
    NUM_JOINTS = 20
    COORDS_PER_JOINT = 3
    DERIVATIVES = 3  # Position, Velocity, Acceleration
    SKELETON_INPUT_DIM = NUM_JOINTS * COORDS_PER_JOINT * DERIVATIVES  # 180

    # Total Input Dimension (Audio + Skeleton)
    INPUT_DIM = N_MFCC + SKELETON_INPUT_DIM  # 13 + 180 = 193

    # Labels
    NUM_CLASSES = 21  # 20 Gestures + 1 Background
    BACKGROUND_LABEL = 0

    # ==========================================
    # Model Architecture
    # ==========================================
    # Stage 1: Gated High-Capacity Kinematic Encoder
    ENCODER_HIDDEN_DIM = 256  # Bi-GRU (128 units per direction * 2)
    ENCODER_LAYERS = 1

    # Stage 2 & 3: Monotonic MS-TCN Refinement
    MSTCN_STAGES = 2  # Number of refinement stages
    MSTCN_LAYERS = 5  # Number of layers per stage
    MSTCN_HIDDEN_DIM = 256
    MSTCN_KERNEL_SIZE = 3
    MSTCN_DILATION_SCHEDULE = [1, 2, 4, 8, 16]  # Monotonic schedule

    DROPOUT = 0.5

    # ==========================================
    # Training Parameters
    # ==========================================
    BATCH_SIZE = 32
    NUM_EPOCHS = 60
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Loss Function Weights
    LOSS_BG_WEIGHT = 0.2  # Weight for background class in CrossEntropy
    LOSS_SMOOTHING_WEIGHT = 0.15  # Weight for Truncated MSE smoothing loss
    SMOOTHING_THRESHOLD = 1.0  # Threshold for truncated MSE

    # Optimization
    PATIENCE = 10  # Early stopping patience

    # ==========================================
    # Inference & Post-Processing
    # ==========================================
    INFERENCE_OVERLAP = 0.5  # 50% overlap for sliding window inference
    MIN_GESTURE_DURATION = 5  # Minimum frames to keep a gesture segment

    # ==========================================
    # Label Mapping
    # ==========================================
    LABEL_MAP = {
        "vattene": 1,
        "vieniqui": 2,
        "perfetto": 3,
        "furbo": 4,
        "cheduepalle": 5,
        "chevuoi": 6,
        "daccordo": 7,
        "seipazzo": 8,
        "combinato": 9,
        "freganiente": 10,
        "ok": 11,
        "cosatifarei": 12,
        "basta": 13,
        "prendere": 14,
        "noncenepiu": 15,
        "fame": 16,
        "tantotempo": 17,
        "buonissimo": 18,
        "messidaccordo": 19,
        "sonostufo": 20,
    }

    # Inverse map for decoding predictions
    ID_TO_NAME = {v: k for k, v in LABEL_MAP.items()}
