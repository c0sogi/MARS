import os


class Config:
    # -------------------------------------------------------------------------
    # General Configuration
    # -------------------------------------------------------------------------
    SEED = 42
    NUM_WORKERS = 4  # Number of data loading workers

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    # Read-only input directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Writable directories
    WORKING_DIR = "./working/idea_66"
    SUBMISSION_DIR = "./submission"

    # Raw Data Files
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")

    # Metadata Files (Pre-generated)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Files (npy format preferred over pickle)
    CACHE_TRAIN_X = os.path.join(WORKING_DIR, "X_train.npy")
    CACHE_TRAIN_Y = os.path.join(WORKING_DIR, "y_train.npy")
    CACHE_TRAIN_ANGLE = os.path.join(WORKING_DIR, "angle_train.npy")

    CACHE_VAL_X = os.path.join(WORKING_DIR, "X_val.npy")
    CACHE_VAL_Y = os.path.join(WORKING_DIR, "y_val.npy")
    CACHE_VAL_ANGLE = os.path.join(WORKING_DIR, "angle_val.npy")

    CACHE_TEST_X = os.path.join(WORKING_DIR, "X_test.npy")
    CACHE_TEST_ANGLE = os.path.join(WORKING_DIR, "angle_test.npy")
    CACHE_TEST_IDS = os.path.join(WORKING_DIR, "ids_test.npy")

    # -------------------------------------------------------------------------
    # Data Hyperparameters
    # -------------------------------------------------------------------------
    IMAGE_SIZE = 75
    IN_CHANNELS = 3  # HH, HV, Average((HH+HV)/2)
    BATCH_SIZE = 32

    # -------------------------------------------------------------------------
    # Model Architecture (IAMSI-CNN)
    # -------------------------------------------------------------------------
    # Plain CNN Backbone: 4 blocks
    BACKBONE_CHANNELS = [64, 128, 128, 128]

    # Activation
    LEAKY_RELU_SLOPE = 0.1

    # Squeeze-and-Excitation
    SE_REDUCTION = 16

    # Readout & Interaction
    # Feature vector size = (Channels_Stage3 + Channels_Stage4) * 2 (Min+Max) + 1 (Angle)
    # 128*2 + 128*2 + 1 = 513 input to interaction layer
    INTERACTION_HIDDEN_DIM = 256

    # Multi-Sample Dropout
    DROPOUT_RATE = 0.5
    NUM_DROPOUT_SAMPLES = 5

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    N_FOLDS = 5
    EPOCHS = 75
    PATIENCE = 12

    # Optimization
    LEARNING_RATE = 1e-3  # Constant LR
    WEIGHT_DECAY = 1e-4  # L2 Regularization

    @classmethod
    def setup_directories(cls):
        """Creates necessary working and submission directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
