import os


class Config:
    """
    Configuration for the Scale-Aligned Dense-Context Hybrid Network (Idea 16).
    Defines file paths, hyperparameters, and data specifications.
    """

    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_opt"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Cache File Paths (Idea Opt Specific)
    # Using .npz for efficient storage of numpy arrays
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_data_opt_v1.npz")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_data_opt_v1.npz")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_data_opt_v1.npz")

    # Output Paths
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # =========================================================================
    # Data Specifications
    # =========================================================================
    SEQ_LEN = 107
    PRED_LEN = 68

    # Input Feature Dimensions
    # 1. Sequence: One-hot (A, G, C, U) -> 4
    # 2. Structure: One-hot ((, ), .) -> 3
    # 3. Loop Type: One-hot (S, M, I, B, H, E, X) -> 7
    # 4. Partner Identity: One-hot (A, G, C, U) -> 4
    INPUT_CHANNELS = 4 + 3 + 7 + 4  # Total: 18

    # Targets
    NUM_TARGETS = 5
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Indices of targets used for scoring and loss calculation (Masked Optimization)
    # reactivity (0), deg_Mg_pH10 (1), deg_Mg_50C (3)
    SCORED_TARGET_INDICES = [0, 1, 3]

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    # Dense Dilated TCN Backbone
    GROWTH_RATE = 64  # Channel width per block
    KERNEL_SIZE = 3
    DROPOUT = 0.1
    DILATIONS = [1, 2, 4, 8, 16, 32]  # Exponentially increasing dilation rates

    # Latent Gather & RNN
    # Project dense history to this dimension BEFORE gathering (Cite Lesson 00062)
    LATENT_DIM = 32
    # BiGRU Hidden size (Cite Lesson 00038, 00004)
    RNN_HIDDEN_DIM = 32
    RNN_LAYERS = 1

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    EPOCHS = 50
    PATIENCE = 10  # Early stopping patience
    NUM_WORKERS = 2  # Data loader workers
