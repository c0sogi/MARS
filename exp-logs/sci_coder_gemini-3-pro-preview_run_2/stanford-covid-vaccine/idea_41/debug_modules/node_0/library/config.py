import os


class Config:
    # Directory Setup
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_41"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata Paths
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Cache File Paths (Explicit Cache Invalidation for RCR-DN)
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_data_rcr_dn_v1.npz")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_data_rcr_dn_v1.npz")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_data_rcr_dn_v1.npz")

    # Output Paths
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # Data Dimensions
    SEQ_LENGTH = 107
    SCORED_LENGTH = 68

    # Input Features:
    # 4 (Sequence: A,G,C,U)
    # + 3 (Structure: ., (, ))
    # + 7 (Loop Type: S,M,I,B,H,E,X)
    # + 4 (Partner Identity: A,G,C,U)
    NUM_NODE_FEATURES = 4 + 3 + 7 + 4

    NUM_TARGETS = 5
    # Indices for scored columns: reactivity (0), deg_Mg_pH10 (1), deg_Mg_50C (3)
    # Submission order: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    SCORED_COLS_INDICES = [0, 1, 3]

    # Model Architecture Hyperparameters
    HIDDEN_DIM = 64  # Latent Dimension Z
    FEEDBACK_DIM = 32  # Feedback Embedding Dimension
    GROWTH_RATE = 64  # For Dense Dilated TCN Backbone
    KERNEL_SIZE = 3
    DILATIONS = [1, 2, 4, 8, 16, 32]
    N_CYCLES = 2  # Number of recycling iterations
    DROPOUT = 0.1

    # Training Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 10  # Early stopping patience

    # Debugging / Development
    DEBUG = False
    DEBUG_SIZE = 100  # Number of samples to use if DEBUG is True
    NUM_WORKERS = 2  # For data loading
