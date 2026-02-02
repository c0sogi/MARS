import os


class Config:
    # ==========================================
    # Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_15"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Input Files
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")

    # Metadata Files
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Paths
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Checkpoint Paths
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ==========================================
    # Data Parameters
    # ==========================================
    ORIG_HEIGHT = 75
    ORIG_WIDTH = 75

    # Upsampling target (ResNet standard)
    IMG_HEIGHT = 224
    IMG_WIDTH = 224

    # Global Normalization Constants (Derived from Data Analysis)
    # Band 1 (HH)
    BAND1_MIN = -45.5944
    BAND1_MAX = 32.1806

    # Band 2 (HV)
    BAND2_MIN = -45.6555
    BAND2_MAX = 17.8628

    # ==========================================
    # Model Parameters
    # ==========================================
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    NUM_CLASSES = 1  # Binary classification
    DROPOUT_RATE = 0.5

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 32
    NUM_WORKERS = 4

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.01

    # Scheduler (ReduceLROnPlateau)
    SCHEDULER_FACTOR = 0.1
    SCHEDULER_PATIENCE = 3

    # Phase 1: Exploration & Trajectory Extraction
    MAX_EPOCHS_PHASE_1 = 50
    EARLY_STOPPING_PATIENCE = 10

    # SWA Configuration
    SWA_DURATION = 12  # Number of epochs to run SWA at the end
    SWA_LR = 1e-4  # Learning rate during SWA phase

    # ==========================================
    # Augmentation
    # ==========================================
    AUG_ROTATION_RANGE = 20  # Degrees

    # ==========================================
    # Execution Flags
    # ==========================================
    DEBUG = False  # Set to True to use a small subset of data for testing
