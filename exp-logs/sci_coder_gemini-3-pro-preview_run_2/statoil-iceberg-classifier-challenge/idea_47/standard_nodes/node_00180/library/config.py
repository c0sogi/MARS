import os
import torch


class Config:
    # ==========================================
    # 1. SYSTEM & REPRODUCIBILITY
    # ==========================================
    PROJECT_NAME = "PC-WBN_Iceberg_Detection"
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # Efficient data loading for 12 vCPUs

    # ==========================================
    # 2. FILE PATHS
    # ==========================================
    # Base Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working Directory (Artifacts & Cache)
    WORK_DIR = "./working/idea_47"
    os.makedirs(WORK_DIR, exist_ok=True)

    # Submission Directory
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Input Files
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files (Pre-generated)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Files
    CACHE_FILE = os.path.join(WORK_DIR, "processed_data.npz")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 3. DATA CONFIGURATION
    # ==========================================
    IMG_HEIGHT = 75
    IMG_WIDTH = 75
    # Channels: Band 1, Band 2, Mean(B1, B2)
    IN_CHANNELS = 3
    NUM_CLASSES = 1  # Binary output (probability)

    # Global Scaling
    # Note: Stats are computed dynamically from training data,
    # but we flag that global scaling is used.
    USE_GLOBAL_SCALING = True

    # ==========================================
    # 4. MODEL ARCHITECTURE
    # ==========================================
    # Backbone
    BACKBONE_FILTERS = 128  # "Sustained Width Strategy"

    # Dual-Path Readout Dimensions
    READOUT_DIM_CONTEXT = 32  # Spatial Context Path (3x3 Conv)
    # Global Path is GAP -> matches backbone filters * 2

    # Metadata Branch
    META_EMBED_DIM = 32

    # Regularization
    DROPOUT_RATE = 0.5  # High dropout for wide backbone

    # ==========================================
    # 5. TRAINING HYPERPARAMETERS
    # ==========================================
    NUM_FOLDS = 5
    BATCH_SIZE = 64
    NUM_EPOCHS = 100

    # Optimization ("Low and Slow")
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.0  # Using standard Adam, not AdamW

    # Scheduler (ReduceLROnPlateau)
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 4

    # Early Stopping
    PATIENCE = 10

    # Augmentation
    # Rotations: 0, 90, 180, 270 + Horizontal Flip
    AUG_ROTATIONS = [0, 90, 180, 270]
    AUG_HFLIP = True
    AUG_VFLIP = False  # Explicitly excluded
    AUG_MIXUP = False  # Explicitly excluded
