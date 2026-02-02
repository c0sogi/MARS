import os
import torch


class Config:
    # ==========================================
    # File Paths and Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Specific working directory for this strategy (Idea 81) to handle caching
    WORKING_DIR = "./working/idea_81"

    # Ensure the working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata file paths (Parquet format)
    TRAIN_FILE = os.path.join(METADATA_DIR, "train.parquet")
    VAL_FILE = os.path.join(METADATA_DIR, "val.parquet")
    TEST_FILE = os.path.join(METADATA_DIR, "test.parquet")

    # Submission template
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output paths
    SUBMISSION_FILE = "./submission/submission.csv"
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # ==========================================
    # Data Parameters
    # ==========================================
    SEQ_LEN = 107
    PRED_LEN = 68

    # Feature Dimensions
    # 4 Nucleotides (A, G, C, U) + 3 Structure (., (, )) + 7 Loop Types (S, M, I, B, H, E, X)
    INPUT_CHANNELS = 14

    # Targets
    NUM_TARGETS = 5
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # Only these columns are used for the competition metric
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # ==========================================
    # Model Architecture (High-Capacity BiGRU)
    # ==========================================
    # Convolutional Stem
    STEM_KERNEL_SIZE = 3
    STEM_FILTERS = 256

    # Backbone
    # Hidden dim is per direction. Total hidden size = 384 * 2 = 768.
    HIDDEN_DIM = 384
    N_LAYERS = 4

    # Regularization
    # Conservative dropout to preserve weak signals in deep networks
    DROPOUT = 0.1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 50

    # Optimization
    MAX_GRAD_NORM = 1.0
    PATIENCE = 10  # Early stopping patience

    # ==========================================
    # Compute & Reproducibility
    # ==========================================
    NUM_WORKERS = 4
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # Debugging
    # ==========================================
    # Set to True to use a small subset of data for quick pipeline testing
    DEBUG = False
    DEBUG_SUBSET_SIZE = 100
