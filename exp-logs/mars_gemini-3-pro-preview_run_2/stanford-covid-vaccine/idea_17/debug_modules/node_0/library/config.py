import os
import torch


class Config:
    # =========================================================================
    # General Settings
    # =========================================================================
    PROJECT_NAME = "RNA_Degradation_Prediction"
    IDEA_NAME = "idea_17"
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SUBSET_SIZE = 100  # Number of samples to use when DEBUG is True

    # =========================================================================
    # Paths
    # =========================================================================
    # Input Metadata (Generated in previous steps)
    TRAIN_CSV = "./metadata/train.csv"
    VAL_CSV = "./metadata/val.csv"
    TEST_CSV = "./metadata/test.csv"

    # Working Directory for this specific idea
    WORKING_DIR = os.path.join("./working", IDEA_NAME)

    # Cache Files (Explicit versioning to force re-processing if needed)
    CACHE_VERSION = "gated_dense_v1"
    TRAIN_CACHE = os.path.join(WORKING_DIR, f"train_data_{CACHE_VERSION}.npz")
    VAL_CACHE = os.path.join(WORKING_DIR, f"val_data_{CACHE_VERSION}.npz")
    TEST_CACHE = os.path.join(WORKING_DIR, f"test_data_{CACHE_VERSION}.npz")

    # Outputs
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # =========================================================================
    # Data Parameters
    # =========================================================================
    SEQ_LEN = 107
    PRED_LEN = 68

    # Columns in the CSV
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Columns used for the competition metric (MCRMSE)
    # We only train/validate on these to avoid negative transfer from auxiliary targets
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # Inputs: Sequence (4) + Structure (3) + LoopType (7) + PartnerIdentity (4)
    # Note: 4 (A,G,C,U) + 3 (.,(,)) + 7 (S,M,I,B,H,E,X) + 4 (Partner A,G,C,U)
    # Total Input Channels = 18
    INPUT_CHANNELS = 4 + 3 + 7 + 4

    # =========================================================================
    # Model Hyperparameters (Gated Dense-Context Hybrid Network)
    # =========================================================================
    # TCN Backbone
    TCN_GROWTH_RATE = 64
    TCN_KERNEL_SIZE = 3
    TCN_DILATIONS = [1, 2, 4, 8, 16, 32]
    TCN_LAYERS = len(TCN_DILATIONS)
    DROPOUT = 0.1

    # Gated Structural Fusion
    # Latent dimension for the query/key vectors in the fusion mechanism
    LATENT_DIM = 64

    # Global BiGRU
    # Hidden dim is set to LATENT_DIM // 2 so that Bidirectional output is LATENT_DIM
    GRU_HIDDEN_DIM = LATENT_DIM // 2

    # =========================================================================
    # Training Parameters
    # =========================================================================
    BATCH_SIZE = 64  # A100 has 40GB, can handle larger batches
    EPOCHS = 100
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    MAX_GRAD_NORM = 5.0

    # Early Stopping
    PATIENCE = 15

    # Scheduler
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 5

    # Hardware
    NUM_WORKERS = 2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup_directories(cls):
        """Ensures the working directory exists."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        print(f"Working directory set to: {cls.WORKING_DIR}")

    @classmethod
    def get_input_dim(cls):
        return cls.INPUT_CHANNELS
