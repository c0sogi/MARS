import os


class Config:
    """
    Central configuration for the Spatial-Kinematic Attentive Refinement Network (SK-ARN) project.
    """

    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORK_DIR = "./working/idea_13"
    SUBMISSION_DIR = "./submission"

    # Create necessary writable directories
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Cache Files (using .npz for efficient storage of numpy arrays)
    TRAIN_CACHE = os.path.join(WORK_DIR, "train_features.npz")
    VAL_CACHE = os.path.join(WORK_DIR, "val_features.npz")
    TEST_CACHE = os.path.join(WORK_DIR, "test_features.npz")
    STATS_CACHE = os.path.join(WORK_DIR, "stats.npz")  # For normalization stats

    # Model Checkpoint
    BEST_MODEL_PATH = os.path.join(WORK_DIR, "best_model.pth")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data & Feature Configuration
    # ==========================================
    # Windowing
    WINDOW_SIZE = 64
    STRIDE = 32  # 50% overlap for sliding window

    # Audio
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_N_MFCC = 13

    # Skeleton
    NUM_JOINTS = 20
    # Features per joint: RelPos(3) + BoneVec(3) + Vel(3) + Acc(3) = 12
    # Total Skeleton Features: 20 * 12 = 240
    # Total Input Features: 240 + 13 (Audio) = 253
    INPUT_DIM = 253

    # Classes
    # 0: Background, 1-20: Gestures
    NUM_CLASSES = 21
    BACKGROUND_CLASS_ID = 0

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 50

    # Loss Weights
    # Weight for background class to handle imbalance
    BACKGROUND_WEIGHT = 0.2

    # Debugging / Development
    # Set to a small integer (e.g., 100) to limit dataset size for quick debugging
    # Set to None for full training
    DEBUG_MAX_SAMPLES = None

    # ==========================================
    # Model Architecture Hyperparameters
    # ==========================================
    # Stage 1: Bi-GRU
    GRU_HIDDEN_SIZE = 128
    GRU_NUM_LAYERS = 2

    # Stage 2 & 3: TCN / Refinement
    TCN_NUM_CHANNELS = 64
    TCN_KERNEL_SIZE = 3
    TCN_DROPOUT = 0.2

    # ==========================================
    # Vocabulary
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

    # Reverse mapping for decoding
    ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
    ID_TO_LABEL[BACKGROUND_CLASS_ID] = "background"
