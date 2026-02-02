import os
import torch


class Config:
    # =========================================================================
    # Directories and Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    SUBMISSION_DIR = "./submission"

    # Specific directory for this idea iteration to prevent cache collisions
    IDEA_DIR = os.path.join(WORKING_DIR, "idea_83")

    # Ensure necessary writeable directories exist
    os.makedirs(IDEA_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # File Paths
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    MODEL_SAVE_PATH = os.path.join(IDEA_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Configuration
    # =========================================================================
    # Cache version key to invalidate old data if preprocessing logic changes
    CACHE_VERSION = "hc_higfn_v1"

    # Sequence Dimensions
    TOTAL_SEQ_LEN = 107
    SCORED_SEQ_LEN = 68

    # Target Columns
    # All 5 conditions provided in the dataset
    ALL_TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # The 3 conditions actually scored in the competition metric
    SCORED_TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # Indices of the scored columns within the full list (0, 1, 3)
    # Used for masking the loss function
    SCORED_TARGET_INDICES = [0, 1, 3]

    # =========================================================================
    # Model Hyperparameters (HC-HIGFN)
    # =========================================================================
    # Dimensions
    LATENT_DIM = 64  # 'Z' dimension (Main Backbone Output)
    FEEDBACK_DIM = 32  # 'E_fb' dimension (Feedback Module Output)
    RNN_HIDDEN_DIM = 64  # Compact Hidden Size for Final Aggregation

    # Backbone Capacity
    MAIN_GROWTH_RATE = 64  # High capacity for static features
    FB_GROWTH_RATE = 16  # Lightweight for feedback loop

    # Architecture Details
    KERNEL_SIZE = 3
    DROPOUT = 0.1
    DILATIONS = [1, 2, 4, 8, 16, 32]  # Exponential dilation schedule

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    BATCH_SIZE = 16  # Strictly 16 as per "Small Batch Regime"
    LEARNING_RATE = 1e-3
    EPOCHS = 50  # Max epochs (will use early stopping)
    PATIENCE = 10  # Early stopping patience
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP = 1.0

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2
