import os
import torch


class Config:
    # ==========================================
    # Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific idea/experiment
    WORKING_DIR = "./working/idea_5"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    OUTPUT_DIR = os.path.join(WORKING_DIR, "outputs")
    SUBMISSION_DIR = "./submission"

    # Metadata files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Model checkpoints and outputs
    MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Hyperparameters
    # ==========================================
    SEED = 42

    # Windowing
    WINDOW_SIZE = 64  # Approx 3.2 seconds
    STRIDE = 32  # 50% overlap for training generation

    # Features
    NUM_JOINTS = 20
    SKELETON_CHANNELS = 3  # (x, y, z)

    # Audio
    AUDIO_SR = 16000
    N_MFCC = 13

    # Classes: 0 is background, 1-20 are gestures
    NUM_CLASSES = 21
    BACKGROUND_CLASS_ID = 0

    # ==========================================
    # Model Architecture Hyperparameters
    # ==========================================
    # Dual-Stream GRU (Stage 1)
    HIDDEN_SIZE = 128
    GRU_LAYERS = 2
    DROPOUT = 0.3

    # TCN Refinement (Stage 2)
    TCN_NUM_CHANNELS = [64, 64, 64]
    TCN_KERNEL_SIZE = 3
    TCN_DROPOUT = 0.2

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 50
    PATIENCE = 7  # Early stopping patience

    # Loss Weights
    BG_WEIGHT = 0.2  # Weight for background class to handle dominance
    SMOOTHING_LOSS_WEIGHT = 0.15  # Weight for log-space smoothing loss (Stage 2)

    # ==========================================
    # Hardware
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # ==========================================
    # Utilities
    # ==========================================
    @staticmethod
    def setup_directories():
        """Ensure necessary directories exist."""
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Label Map for reference
    LABEL_MAP = {
        1: "vattene",
        2: "vieniqui",
        3: "perfetto",
        4: "furbo",
        5: "cheduepalle",
        6: "chevuoi",
        7: "daccordo",
        8: "seipazzo",
        9: "combinato",
        10: "freganiente",
        11: "ok",
        12: "cosatifarei",
        13: "basta",
        14: "prendere",
        15: "noncenepiu",
        16: "fame",
        17: "tantotempo",
        18: "buonissimo",
        19: "messidaccordo",
        20: "sonostufo",
    }
