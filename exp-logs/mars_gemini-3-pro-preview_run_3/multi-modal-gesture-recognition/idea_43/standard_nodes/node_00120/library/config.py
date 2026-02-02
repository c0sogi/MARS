import os
import torch


class Config:
    """
    Configuration for the Physically-Aligned Moderate-Capacity Network (PAM-CN).
    Defines hyperparameters, file paths, and constants for the gesture recognition pipeline.
    """

    # ==========================================
    # System & Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for Idea 43 (PAM-CN)
    WORKING_DIR = "./working/idea_43"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"

    # Ensure necessary directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # Data Engineering & Physical Alignment
    # ==========================================
    # Deterministic Physical Scaling: Convert millimeters to meters
    # This aligns Position magnitude (~1.0) with Audio/Normalized features
    PHYSICAL_SCALE = 0.001

    # Skeleton Configuration
    NUM_JOINTS = 20
    # Features per joint: Position(3) + Velocity(3) + Acceleration(3)
    CHANNELS_PER_JOINT = 9
    SKELETON_INPUT_DIM = NUM_JOINTS * CHANNELS_PER_JOINT  # 180 features

    # Audio Configuration
    N_MFCC = 13
    AUDIO_INPUT_DIM = N_MFCC

    # Total Input Dimension for Early Fusion
    INPUT_DIM = SKELETON_INPUT_DIM + AUDIO_INPUT_DIM  # 193 features

    # Sampling Strategy
    WINDOW_SIZE = 64
    STRIDE = 32

    # ==========================================
    # Model Architecture (PAM-CN)
    # ==========================================
    # Stage 1: Kinematic Encoder (Bi-GRU)
    # 96 units per direction = 192 total
    HIDDEN_DIM = 192
    DROPOUT_ENCODER = 0.4

    # Stage 2 & 3: Monotonic Non-Causal Refinement (TCN)
    # Input to TCN is strictly class probabilities (dim=NUM_CLASSES)
    TCN_HIDDEN_DIM = 64
    TCN_KERNEL_SIZE = 3
    # Monotonically increasing dilations for receptive field = 63 < 64
    DILATIONS = [1, 2, 4, 8, 16]
    DROPOUT_TCN = 0.2

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    NUM_CLASSES = 21  # 20 Gestures + 1 Background
    BACKGROUND_CLASS_ID = 0

    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10

    # Loss Function Configuration
    # Weighted Cross Entropy: Down-weight background to focus on gestures
    CLASS_WEIGHTS = torch.ones(NUM_CLASSES)
    CLASS_WEIGHTS[BACKGROUND_CLASS_ID] = 0.2

    # Log-Space Smoothing Loss (Truncated MSE)
    SMOOTHING_LAMBDA = 0.15
    SMOOTHING_THRESHOLD = 1.0

    # ==========================================
    # Inference & Post-Processing
    # ==========================================
    # Minimum duration in frames to be considered a valid gesture
    MIN_GESTURE_LENGTH = 5

    # ==========================================
    # Label Map
    # ==========================================
    # Maps gesture names to IDs (1-20). ID 0 is reserved for background.
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

    # Reverse mapping for decoding predictions
    ID_TO_NAME = {v: k for k, v in LABEL_MAP.items()}
