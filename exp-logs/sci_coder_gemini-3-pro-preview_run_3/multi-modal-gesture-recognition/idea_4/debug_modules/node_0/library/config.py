import os


class Config:
    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_4"
    SUBMISSION_DIR = "./submission"

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Cache Files (using .npz for efficient storage)
    TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_features.npz")
    VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_features.npz")
    TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_features.npz")

    # Model Checkpoint
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Submission Output
    SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Create directories if they don't exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Data Processing Hyperparameters
    # ==========================================
    # Windowing Strategy
    WINDOW_SIZE = 64
    STRIDE = 32  # Overlap for generating training samples

    # Feature Engineering
    # Skeleton: 20 joints * 3 coordinates (x, y, z)
    NUM_JOINTS = 20
    COORDS_PER_JOINT = 3
    SKELETON_RAW_DIM = NUM_JOINTS * COORDS_PER_JOINT  # 60

    # Derivatives
    USE_VELOCITY = True  # 1st derivative
    USE_ACCELERATION = True  # 2nd derivative

    # Audio
    AUDIO_N_MFCC = 13

    # Total Input Dimension Calculation
    # Structure: [Pos (60), Vel (60), Acc (60), Audio (13)]
    INPUT_DIM = (
        SKELETON_RAW_DIM * (1 + int(USE_VELOCITY) + int(USE_ACCELERATION))
        + AUDIO_N_MFCC
    )

    # ==========================================
    # Model Architecture
    # ==========================================
    NUM_CLASSES = 21  # 20 Gestures + 1 Background (Index 0)

    # Stage 1: Contextual Encoder (Bi-GRU)
    GRU_HIDDEN_SIZE = 128
    GRU_NUM_LAYERS = 2
    GRU_DROPOUT = 0.3

    # Stage 2: Holo-Refinement Module (TCN)
    # Input: Concatenation of Stage 1 Logits + Stage 1 Latent Features (Bi-Directional)
    # Latent Feature Dim = GRU_HIDDEN_SIZE * 2
    REFINEMENT_INPUT_DIM = NUM_CLASSES + (GRU_HIDDEN_SIZE * 2)

    TCN_CHANNELS = [128, 128, 128, 128]
    TCN_KERNEL_SIZE = 3
    TCN_DROPOUT = 0.2

    # ==========================================
    # Training Configuration
    # ==========================================
    NUM_EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_PATIENCE = 10

    # Loss Function Weights
    # Background class (index 0) is weighted 0.2, others 1.0
    CLASS_WEIGHTS = [0.2] + [1.0] * 20

    # Log-Space Smoothing Loss Weight
    SMOOTHING_LAMBDA = 0.15

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

    # Helper to invert map if needed
    ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
