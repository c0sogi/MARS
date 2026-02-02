import os
import torch


class Config:
    # ==========================================
    # 1. PATHS & DIRECTORIES
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for artifacts (idea_39 based on prompt context)
    # This directory is used for caching processed data and saving models
    WORK_DIR = "./working/idea_39"
    SUBMISSION_DIR = "./submission"

    # Ensure output directories exist
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Input Files
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")

    # Metadata Files (Pre-generated)
    TRAIN_META_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Cache File (for deterministic data processing)
    CACHE_FILE = os.path.join(WORK_DIR, "processed_data.npz")

    # Submission Output
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 2. DATA HYPERPARAMETERS
    # ==========================================
    IMG_WIDTH = 75
    IMG_HEIGHT = 75
    # 3 Channels: Band 1 (HH), Band 2 (HV), Mean ((HH+HV)/2)
    CHANNELS = 3

    # Normalization Strategy
    # Use global min-max statistics from the full training set
    # (Prevents covariate shift from fold-wise scaling)
    USE_GLOBAL_SCALING = True

    # ==========================================
    # 3. AUGMENTATION SETTINGS
    # ==========================================
    # Rotations: 0, 90, 180, 270 degrees
    AUG_ROTATION = True
    # Horizontal Flip
    AUG_H_FLIP = True
    # Vertical Flip (Disabled to avoid unrealistic sensor orientation artifacts)
    AUG_V_FLIP = False
    # Mixup (Disabled to preserve sharp edge features)
    AUG_MIXUP = False

    # ==========================================
    # 4. MODEL ARCHITECTURE (DM-WBN)
    # ==========================================
    # Wide-Body Backbone Filters (Sustained Width Strategy)
    BACKBONE_FILTERS = 128

    # Attention Mechanism (CBAM before pooling)
    USE_CBAM = True

    # Pooling Strategy
    # Dual-Pooling: Concatenate MaxPool (Peaks) and MinPool (Shadows)
    USE_DUAL_POOLING = True

    # Readout Strategy
    # Decoupled Morphological Readout (Split Peak vs Shadow paths)
    DECOUPLED_READOUT = True
    READOUT_DIM = 32  # Dimension for the decoupled spatial features before fusion

    # Regularization
    # High dropout to regularize the wide backbone
    DROPOUT_RATE = 0.5

    # ==========================================
    # 5. TRAINING HYPERPARAMETERS
    # ==========================================
    SEED = 42
    N_FOLDS = 5

    # Batch Size & Epochs
    BATCH_SIZE = 32
    EPOCHS = 50

    # Optimization
    # Adam Optimizer settings
    LEARNING_RATE = 1e-3
    # Only introduce weight decay if validation loss consistently exceeds training loss
    WEIGHT_DECAY = 0.0

    # Scheduler (ReduceLROnPlateau)
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 3

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 10

    # ==========================================
    # 6. HARDWARE
    # ==========================================
    NUM_WORKERS = 4  # Optimized for 12 vCPUs
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
