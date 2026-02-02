import os


class Config:
    # ==========================================
    # File Paths
    # ==========================================
    # Input Metadata (generated in previous steps)
    TRAIN_DATA_PATH = "./metadata/train.parquet"
    VAL_DATA_PATH = "./metadata/val.parquet"
    TEST_DATA_PATH = "./metadata/test.parquet"

    # Raw Input
    SAMPLE_SUBMISSION_PATH = "./input/sample_submission.csv"

    # Output Directories
    WORKING_DIR = "./working/idea_1"
    SUBMISSION_DIR = "./submission"

    # Ensure output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Cache File Paths
    TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_features.npy")
    TRAIN_TARGETS_PATH = os.path.join(WORKING_DIR, "train_targets.npy")
    VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_features.npy")
    VAL_TARGETS_PATH = os.path.join(WORKING_DIR, "val_targets.npy")
    TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_features.npy")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "mlp_model.pth")

    # ==========================================
    # Data Constants
    # ==========================================
    SEQ_LENGTH = 107
    SEQ_SCORED = 68
    NUM_TARGETS = 5  # reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # ==========================================
    # Feature Mappings (Tokenization)
    # ==========================================
    # Nucleotide Sequence
    TOKEN_MAP_SEQ = {"A": 0, "G": 1, "C": 2, "U": 3}

    # Secondary Structure (Dot-Bracket)
    TOKEN_MAP_STRUCT = {"(": 0, ")": 1, ".": 2}

    # Predicted Loop Type
    TOKEN_MAP_LOOP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    # Vocabulary Sizes
    VOCAB_SIZE_SEQ = len(TOKEN_MAP_SEQ)
    VOCAB_SIZE_STRUCT = len(TOKEN_MAP_STRUCT)
    VOCAB_SIZE_LOOP = len(TOKEN_MAP_LOOP)

    # Total channels per position after one-hot encoding
    # 4 (Seq) + 3 (Struct) + 7 (Loop) = 14
    CHANNELS_PER_POS = VOCAB_SIZE_SEQ + VOCAB_SIZE_STRUCT + VOCAB_SIZE_LOOP

    # Flattened Input Dimension: 107 * 14 = 1498
    INPUT_DIM = SEQ_LENGTH * CHANNELS_PER_POS

    # Output Dimension: 68 * 5 = 340
    OUTPUT_DIM = SEQ_SCORED * NUM_TARGETS

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 64
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Model Architecture
    HIDDEN_LAYERS = [512, 256, 256]
    DROPOUT_RATE = 0.5

    # RNN Architecture
    RNN_HIDDEN_DIM = 256
    RNN_LAYERS = 2
    RNN_DROPOUT = 0.5

    # Early Stopping
    PATIENCE = 10
