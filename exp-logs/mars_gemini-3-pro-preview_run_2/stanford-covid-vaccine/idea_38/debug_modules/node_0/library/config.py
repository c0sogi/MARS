import os
import torch


class Config:
    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2  # Number of data loading workers

    # =========================================================================
    # Data Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Working directory specific to Idea 38
    WORKING_DIR = "./working/idea_38"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata Files (Generated previously)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Cache Files (Numpy format for speed)
    # Using specific names for this idea to avoid conflicts
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_data_df_dcn_v1.npz")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_data_df_dcn_v1.npz")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_data_df_dcn_v1.npz")

    # Output Paths
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # =========================================================================
    # Data Dimensions
    # =========================================================================
    SEQ_LENGTH = 107
    SEQ_SCORED = 68

    # Target columns used for scoring
    # Note: Training uses these 3 for loss, but submission requires all 5 columns (others are 0 or ignored)
    SCORED_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    ALL_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    NUM_TARGETS = 5  # Total output channels required

    # =========================================================================
    # Model Hyperparameters (DF-DCN)
    # =========================================================================
    # Main Backbone (Static Dense TCN)
    MAIN_GROWTH_RATE = 64
    MAIN_LAYERS = 6  # Dilations: 1, 2, 4, 8, 16, 32
    MAIN_KERNEL_SIZE = 3
    MAIN_LATENT_DIM = 64  # Z dimension

    # Feedback Backbone (Dynamic Dense TCN)
    FB_GROWTH_RATE = 32
    FB_LAYERS = 4  # Dilations: 1, 2, 4, 8
    FB_KERNEL_SIZE = 3
    FB_LATENT_DIM = 32  # E_fb dimension

    # RNN Aggregator
    # Input to RNN is (Z + E_fb) * 2 (Self + Partner) = (64 + 32) * 2 = 192
    # Hidden size is input // 2 = 96
    RNN_HIDDEN_SIZE = 96
    RNN_LAYERS = 1
    RNN_DROPOUT = 0.0  # Only relevant if layers > 1

    DROPOUT = 0.1

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 32
    EPOCHS = 25
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Scheduler
    PATIENCE = 3
    FACTOR = 0.5
    MIN_LR = 1e-6

    # Early Stopping
    ES_PATIENCE = 6

    # Recycling
    RECYCLE_PASSES = 2  # Pass 1 (Zero Init), Pass 2 (Feedback)
    LOSS_WEIGHT_PASS_1 = 0.5
    LOSS_WEIGHT_PASS_2 = 1.0

    # Debugging
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SUBSET_SIZE = 100
