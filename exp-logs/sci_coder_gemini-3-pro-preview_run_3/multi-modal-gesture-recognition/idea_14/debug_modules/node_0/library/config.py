import os
import torch


class Config:
    # ==========================================
    # Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific idea
    WORKING_DIR = "./working/idea_14"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # Data Processing Hyperparameters
    # ==========================================
    # Windowing
    WINDOW_SIZE = 64
    STRIDE = 32  # 50% overlap

    # Skeleton Structure
    NUM_JOINTS = 20
    COORDS_PER_JOINT = 3  # X, Y, Z

    # Feature Flags
    USE_REL_POS = True  # Root-relative positions
    USE_BONE_VEC = True  # Bone vectors (spatial derivative)
    USE_VELOCITY = True  # 1st temporal derivative
    USE_ACCEL = True  # 2nd temporal derivative

    # Audio
    AUDIO_MFCC_N_MFCC = 13
    AUDIO_SAMPLE_RATE = 16000  # Target sample rate

    # Input Dimension Calculation
    # Base: 20 joints * 3 coords = 60
    # Rel Pos (60) + Bone Vec (60) + Vel (60) + Accel (60) + Audio (13) = 253
    INPUT_DIM = (NUM_JOINTS * COORDS_PER_JOINT * 4) + AUDIO_MFCC_N_MFCC

    # Labels
    # 0 = Background, 1-20 = Gestures
    NUM_CLASSES = 21

    # ==========================================
    # Model Architecture
    # ==========================================
    # Stage 1: Bi-GRU Encoder
    GRU_HIDDEN_DIM = 128
    GRU_NUM_LAYERS = 2
    DROPOUT = 0.3

    # Stage 2 & 3: MSTCN / Refinement
    TCN_NUM_CHANNELS = 64
    TCN_KERNEL_SIZE = 3
    TCN_NUM_LAYERS = 10  # Number of dilated layers per stage

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10

    # Loss Weights
    # Weight 0.2 for background (class 0), 1.0 for others
    CLASS_WEIGHTS = [0.2] + [1.0] * 20

    # Smoothing Loss Weight (Truncated MSE)
    LAMBDA_SMOOTHING = 0.15

    # ==========================================
    # Hardware
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    @classmethod
    def setup(cls):
        """Ensures necessary directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Run setup immediately when module is imported to guarantee paths exist
Config.setup()
