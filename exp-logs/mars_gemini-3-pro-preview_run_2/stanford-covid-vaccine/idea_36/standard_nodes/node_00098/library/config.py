import os
import torch


class Config:
    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Specific working directory for Idea 36 to ensure cache isolation
    WORKING_DIR = "./working/idea_36"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # File paths
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Cache file names
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_data_cf_dcn_v1.npz")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_data_cf_dcn_v1.npz")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_data_cf_dcn_v1.npz")

    # Model Checkpoint
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # =========================================================================
    # Data Specifications
    # =========================================================================
    SEQ_LENGTH = 107
    SEQ_SCORED = 68

    # Input features:
    # 4 (Sequence: A,G,C,U) +
    # 3 (Structure: (,.,)) +
    # 7 (LoopType: S,M,I,B,H,E,X) +
    # 4 (Partner Identity: A,G,C,U - Explicit injection)
    INPUT_CHANNELS = 4 + 3 + 7 + 4

    # Targets
    NUM_TARGETS = 5
    SCORED_TARGET_INDICES = [0, 1, 3]  # reactivity, deg_Mg_pH10, deg_Mg_50C

    # =========================================================================
    # Model Architecture (CF-DCN)
    # =========================================================================
    # Backbone (Static Dense Dilated TCN)
    EMBED_DIM = 64
    GROWTH_RATE = 64
    DROPOUT = 0.1
    KERNEL_SIZE = 3
    DILATIONS = [1, 2, 4, 8, 16, 32]

    # Contextualized Feedback Module
    FEEDBACK_DIM = 32  # Dimension for the lightweight feedback TCN

    # RNN Aggregation
    RNN_HIDDEN_DIM = EMBED_DIM // 2  # Bidirectional, so output is EMBED_DIM
    RNN_LAYERS = 1

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 50
    PATIENCE = 5  # Early stopping patience

    # Loss weighting
    # L_total = MCRMSE(Y_final) + AUX_WEIGHT * MCRMSE(Y_aux)
    AUX_WEIGHT = 0.5

    # =========================================================================
    # Compute
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2

    # =========================================================================
    # Debugging
    # =========================================================================
    DEBUG = False
    DEBUG_SIZE = 100  # Number of samples to use if DEBUG is True
