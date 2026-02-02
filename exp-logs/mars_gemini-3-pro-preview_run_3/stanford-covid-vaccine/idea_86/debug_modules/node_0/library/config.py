import os


class Config:
    # ==========================================
    # Paths and Directories
    # ==========================================
    # Input Metadata (Parquet files)
    TRAIN_DATA_PATH = "./metadata/train.parquet"
    VAL_DATA_PATH = "./metadata/val.parquet"
    TEST_DATA_PATH = "./metadata/test.parquet"

    # Working Directory for this specific idea/run
    WORKING_DIR = "./working/idea_86"

    # Cache Directory for processed tensors
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Model Checkpoint Path
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Final Submission Path
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Specifications
    # ==========================================
    SEQ_LEN = 107
    SEQ_SCORED = 68

    # Input Features:
    # 4 (A,G,C,U) + 3 (.,(,)) + 7 (Loop Types) = 14
    INPUT_DIM = 14

    # Target Columns for Training (Multi-Task Learning)
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Target Columns for Scoring/Validation
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # ==========================================
    # Model Hyperparameters
    # Strategy: High-Capacity Stabilized GLU-Decoupled BiGRU
    # ==========================================
    # Backbone
    HIDDEN_DIM = 384  # Per direction. Bidirectional total = 768.
    NUM_LAYERS = 4  # Deep 4-layer backbone

    # Convolutional Stem
    CONV_FILTERS = 256
    KERNEL_SIZE = 3

    # Regularization
    DROPOUT = 0.1  # Conservative dropout

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Optimization Stability
    GRAD_CLIP = 1.0  # Mandatory for stability

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 5

    # ==========================================
    # Debugging / Development
    # ==========================================
    # Set DEBUG to True to train on a small subset for quick verification
    DEBUG = False
    DEBUG_SUBSET_SIZE = 100

    @staticmethod
    def setup():
        """Ensure all necessary working directories exist."""
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
