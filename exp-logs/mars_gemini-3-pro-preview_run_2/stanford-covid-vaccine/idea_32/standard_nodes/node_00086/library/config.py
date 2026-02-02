import os


class Config:
    """
    Configuration for the Normalized Recurrent Dense-Context Network (NR-DCN).
    """

    # --------------------------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------------------------
    SEED = 42

    # --------------------------------------------------------------------------
    # File Paths & Directories
    # --------------------------------------------------------------------------
    # Metadata directories (Read-Only)
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Working directory (Write Access)
    WORKING_DIR = "./working/idea_32"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Cache Files for Processed Tensors
    # Explicitly named to ensure cache invalidation from previous ideas
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_data_normalized_recurrent_v1.npz")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_data_normalized_recurrent_v1.npz")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_data_normalized_recurrent_v1.npz")

    # Model Checkpoint & Submission
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Specifications
    # --------------------------------------------------------------------------
    SEQ_LEN = 107
    PRED_LEN = 68

    # Target Columns
    # All 5 are predicted, but only 3 are scored in the metric
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # Input Feature Dimensions
    # 4 (Sequence: A,G,C,U)
    # + 3 (Structure: (,.,))
    # + 7 (Loop Type: S,M,I,B,H,E,X)
    # + 4 (Partner Identity: A,G,C,U)
    # + 5 (Recycling Channels: prev_preds)
    INPUT_CHANNELS = 4 + 3 + 7 + 4 + 5  # Total: 23

    # --------------------------------------------------------------------------
    # Model Hyperparameters
    # --------------------------------------------------------------------------
    # Dense TCN Backbone
    GROWTH_RATE = 64
    DROPOUT = 0.1
    KERNEL_SIZE = 3
    DILATIONS = [1, 2, 4, 8, 16, 32]  # Exponential dilation for global context

    # Structural Interaction
    LATENT_DIM = 64  # Dimension to project to before gathering partner features

    # Global Aggregation (BiGRU)
    GRU_HIDDEN_DIM = 64  # Hidden size for the BiGRU

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    EPOCHS = 50
    PATIENCE = 10  # For Early Stopping
    NUM_WORKERS = 2

    # --------------------------------------------------------------------------
    # Debugging
    # --------------------------------------------------------------------------
    DEBUG = False
    DEBUG_SUBSET_SIZE = 100
