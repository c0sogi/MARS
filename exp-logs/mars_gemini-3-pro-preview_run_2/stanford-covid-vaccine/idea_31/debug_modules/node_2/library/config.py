import os


class Config:
    # =========================================================================
    # General Configuration
    # =========================================================================
    SEED = 42
    NUM_WORKERS = 2  # Adjusted for the available vCPUs

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory specific to this idea iteration
    WORKING_DIR = "./working/idea_31"

    # Create working directory immediately
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Cache Files - Explicit naming to force fresh generation for SR-DCN
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_data_sr_dcn_v1.npz")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_data_sr_dcn_v1.npz")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_data_sr_dcn_v1.npz")

    # Submission output
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Specifications
    # =========================================================================
    SEQ_LEN = 107
    SCORED_LEN = 68

    # All 5 targets are predicted and used for recycling
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Only these 3 are used for the primary metric calculation
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # =========================================================================
    # Model Hyperparameters (SR-DCN)
    # =========================================================================
    # Input
    RECYCLING_CHANNELS = 5  # Number of channels for the feedback loop (predictions)

    # Backbone (Dense Dilated TCN)
    GROWTH_RATE = 64  # Number of filters added per dense block
    KERNEL_SIZE = 3
    DILATIONS = [1, 2, 4, 8, 16, 32]  # Exponential dilation for global context
    DROPOUT = 0.1

    # Latent Structural Interaction
    HIDDEN_DIM = 64  # Dimension for projection and partner gathering
    # Also used as hidden size for the BiGRU (input_dim // 2)

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 16
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Loss weights
    AUX_LOSS_WEIGHT = 0.5  # Weight for the loss on the first pass (Cold Start)

    # Scheduler / Early Stopping
    PATIENCE = 5
    FACTOR = 0.5  # Learning rate reduction factor
