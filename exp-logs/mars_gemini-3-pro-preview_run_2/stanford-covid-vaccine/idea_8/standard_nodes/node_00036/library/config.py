import os


class Config:
    # =========================================================================
    # File Paths
    # =========================================================================
    # Input Data
    TRAIN_DATA_PATH = "./metadata/train.csv"
    VAL_DATA_PATH = "./metadata/val.csv"
    TEST_DATA_PATH = "./metadata/test.csv"
    SAMPLE_SUBMISSION_PATH = "./input/sample_submission.csv"

    # Output / Working Directory
    # We use idea_8 as the designated working folder for this iteration
    WORKING_DIR = "./working/idea_8/"
    os.makedirs(WORKING_DIR, exist_ok=True)

    CACHE_DIR = WORKING_DIR  # Cache processed numpy files here
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # =========================================================================
    # Data Parameters
    # =========================================================================
    SEQ_LENGTH = 107
    SEQ_SCORED = 68

    # Vocabulary Sizes for One-Hot Encoding
    VOCAB_SIZE_SEQ = 4  # A, G, C, U
    VOCAB_SIZE_STRUCT = 3  # (, ), .
    VOCAB_SIZE_LOOP = 7  # S, M, I, B, H, E, X

    # Partner Identity Feature Size (Same as Seq Vocab)
    PARTNER_FEAT_SIZE = 4

    # All available target columns in the dataset
    ALL_TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # The specific targets used for the competition metric
    SCORED_TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # Indices of the scored targets within the ALL_TARGET_COLS list
    # reactivity (0), deg_Mg_pH10 (1), deg_pH10 (2), deg_Mg_50C (3), deg_50C (4)
    # Selected: [0, 1, 3]
    SCORED_INDICES = [0, 1, 3]

    NUM_TARGETS = len(ALL_TARGET_COLS)

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    # Input Dimension: Seq(4) + Struct(3) + Loop(7) + PartnerID(4) = 18
    INPUT_CHANNELS = (
        VOCAB_SIZE_SEQ + VOCAB_SIZE_STRUCT + VOCAB_SIZE_LOOP + PARTNER_FEAT_SIZE
    )

    # Dense Dilated TCN Backbone
    TCN_CHANNELS = 64  # Output channels for each TCN block
    TCN_KERNEL_SIZE = 3
    TCN_LAYERS = 6  # Number of layers (dilations: 1, 2, 4, 8, 16, 32)

    # Latent Structural Interaction
    LATENT_DIM = 32  # Dimension to compress to before gathering partner features

    # Global Aggregation (BiGRU)
    # Hidden size is set to TCN_CHANNELS // 2 so bidirectional output is TCN_CHANNELS
    GRU_HIDDEN_DIM = TCN_CHANNELS // 2

    DROPOUT = 0.1

    # =========================================================================
    # Training Parameters
    # =========================================================================
    SEED = 42
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    EPOCHS = 50
    NUM_WORKERS = 2

    # Early Stopping
    PATIENCE = 10

    # Debugging / Development
    # Set to True to use a smaller subset of data for rapid testing
    DEBUG = False
    DEBUG_SUBSET_SIZE = 100
