import os


class Config:
    # =========================================================================
    # Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Using idea_29 as the specific working directory for this iteration
    WORK_DIR = "./working/idea_29"

    # Raw Data
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata
    TRAIN_META = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Outputs
    CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORK_DIR, "submission")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Create directories
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Global Statistics (Derived from Data Analysis)
    # =========================================================================
    # Band 1 (HH)
    BAND1_MIN = -45.5944
    BAND1_MAX = 32.1806

    # Band 2 (HV)
    BAND2_MIN = -45.6555
    BAND2_MAX = 17.8628

    # Incidence Angle
    INC_ANGLE_MEAN = 39.2829
    INC_ANGLE_STD = 3.8362

    # Missing Angle Imputation Value (if needed, though we use mean usually)
    INC_ANGLE_FILL = INC_ANGLE_MEAN

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    SEED = 42
    IMG_SIZE = 224  # Upsampling from 75x75
    IN_CHANNELS = 3  # Band1, Band2, Mean(Band1, Band2)
    NUM_CLASSES = 1  # Binary classification

    # Architecture
    BACKBONE = "resnet18"
    DROPOUT_RATE = 0.5
    USE_LATE_FUSION = True

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 32
    NUM_WORKERS = 4

    # Optimizer
    LR_BASE = 2e-4
    WEIGHT_DECAY = 0.01

    # Loss
    LABEL_SMOOTHING = 0.05

    # Phase 1: Calibration (ReduceLROnPlateau)
    PHASE1_MAX_EPOCHS = 50  # Sufficient headroom for early stopping
    PHASE1_PATIENCE = 10
    PHASE1_FACTOR = 0.5

    # Phase 2: Production (Cosine + SWA)
    # Note: The actual T_max for Cosine is determined dynamically from Phase 1 result
    LR_SWA = 1e-5
    SWA_EPOCHS = 12

    # Cross Validation
    NUM_FOLDS = 5
