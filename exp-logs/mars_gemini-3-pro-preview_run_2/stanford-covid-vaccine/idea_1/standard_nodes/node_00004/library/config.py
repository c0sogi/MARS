import os


class Config:
    """
    Configuration class for the RNA degradation prediction task.
    Contains file paths, data constants, mappings, and hyperparameters.
    """

    # =========================================================================
    # File Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    SUBMISSION_DIR = "./submission"

    # Input Files
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files (Stratified Splits)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Cache Directory for processed data
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_1")

    # =========================================================================
    # Data Dimensions & Constants
    # =========================================================================
    SEQ_LEN = 107
    SCORED_LEN = 68

    # =========================================================================
    # Mappings (Vocabularies)
    # =========================================================================
    # Sequence: A, G, U, C
    TOKEN2INT_SEQ = {"A": 0, "G": 1, "C": 2, "U": 3}

    # Structure: (, ), .
    TOKEN2INT_STRUCT = {"(": 0, ")": 1, ".": 2}

    # Predicted Loop Type: S, M, I, B, H, E, X
    TOKEN2INT_LOOP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    # =========================================================================
    # Input Feature Configuration
    # =========================================================================
    NUM_SEQ_TOKENS = len(TOKEN2INT_SEQ)  # 4
    NUM_STRUCT_TOKENS = len(TOKEN2INT_STRUCT)  # 3
    NUM_LOOP_TOKENS = len(TOKEN2INT_LOOP)  # 7

    # Total input channels for One-Hot Encoding
    # 4 (Seq) + 3 (Struct) + 7 (Loop) = 14
    INPUT_CHANNELS = NUM_SEQ_TOKENS + NUM_STRUCT_TOKENS + NUM_LOOP_TOKENS

    # =========================================================================
    # Target Columns
    # =========================================================================
    # All ground truth columns provided in training data
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Columns specifically used for the competition metric (MCRMSE)
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    NUM_TARGETS = len(TARGET_COLS)

    # =========================================================================
    # Model Hyperparameters (Dilated ResNet)
    # =========================================================================
    HIDDEN_DIM = 256
    KERNEL_SIZE = 3
    DROPOUT = 0.1
    # Exponential dilation rates to capture long-range dependencies
    DILATIONS = [1, 2, 4, 8, 16, 32]

    # =========================================================================
    # Training Settings
    # =========================================================================
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    EPOCHS = 35
    NUM_WORKERS = 4
    SEED = 42

    # Early Stopping
    PATIENCE = 10

    @classmethod
    def create_dirs(cls):
        """Ensures necessary working directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
