import os
import torch


class Config:
    """
    Configuration for the Normalized Gated-Kinematic Refinement Network (NG-KRN).
    Idea 29: Robust Gated High-Capacity Monotonic Network with Modality-Specific Normalization.
    """

    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for Idea 29
    WORKING_DIR = "./working/idea_29"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata Files (Pre-generated)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Stats File for Modality-Specific Normalization
    STATS_FILE = os.path.join(CACHE_DIR, "stats.npz")

    # Ensure critical directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # Debugging Flags
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 20  # Subset size for debugging

    # ==========================================
    # Data Pipeline Hyperparameters
    # ==========================================
    # Sliding Window Strategy
    WINDOW_SIZE = 64
    STRIDE_TRAIN = 32
    STRIDE_TEST = 32  # 50% overlap for temporal ensembling

    # Feature Dimensions
    # Skeleton: 20 joints * 3 coords (x,y,z) * 3 derivatives (pos, vel, acc)
    NUM_SKELETON_JOINTS = 20
    SKELETON_CHANNELS = 9  # 3 (Pos) + 3 (Vel) + 3 (Acc)
    SKELETON_INPUT_DIM = NUM_SKELETON_JOINTS * SKELETON_CHANNELS  # 180

    # Audio: MFCCs
    AUDIO_MFCC_N_MFCC = 13
    AUDIO_INPUT_DIM = AUDIO_MFCC_N_MFCC

    # Total Early Fusion Input Dimension
    INPUT_DIM = SKELETON_INPUT_DIM + AUDIO_INPUT_DIM  # 193

    # ==========================================
    # Model Architecture (NG-KRN)
    # ==========================================
    # Stage 1: Normalized Gated Kinematic Encoder (Bi-GRU)
    GRU_HIDDEN_SIZE = 128  # Per direction
    GRU_NUM_LAYERS = 2
    GRU_DROPOUT = 0.3
    GRU_BIDIRECTIONAL = True
    # Total Hidden Dimension = 128 * 2 = 256
    HIDDEN_DIM = 256

    # Stage 2 & 3: Monotonic Non-Causal MS-TCN
    TCN_NUM_CHANNELS = 256
    TCN_KERNEL_SIZE = 3
    # Monotonically Increasing Dilation Schedule
    TCN_DILATIONS = [1, 2, 4, 8, 16]
    TCN_DROPOUT = 0.3

    # Output Classes: 20 Gestures + 1 Background
    NUM_CLASSES = 21

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    NUM_EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Loss Configuration
    # Background (Class 0) Weight = 0.2
    CLASS_WEIGHTS = [0.2] + [1.0] * 20

    # Log-Space Smoothing Loss (Truncated MSE)
    SMOOTHING_LAMBDA = 0.15
    SMOOTHING_THRESHOLD = 1.0

    # Optimization
    EARLY_STOPPING_PATIENCE = 10

    # ==========================================
    # Inference / Post-Processing
    # ==========================================
    # Minimum duration to keep a gesture prediction
    MIN_DURATION_FRAMES = 5

    # ==========================================
    # Vocabulary
    # ==========================================
    # Mapping: Gesture Name -> ID (1-20). ID 0 is Background.
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

    # Reverse Mapping for Submission
    ID_TO_NAME = {v: k for k, v in LABEL_MAP.items()}
    ID_TO_NAME[0] = "background"
