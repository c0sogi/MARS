import os
import torch


class Config:
    # ==========================================
    # Path Configuration
    # ==========================================
    # Root directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_2"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Raw Data Files
    TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
    TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Cache Files (for deterministic data processing)
    # Using .npy / .npz for efficient storage of processed arrays
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_data.npz")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_data.npz")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_data.npz")
    SCALER_CACHE = os.path.join(WORKING_DIR, "scaler_params.npz")

    # Model Checkpoint
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # ==========================================
    # Data & Feature Configuration
    # ==========================================
    # Random Seed for reproducibility
    SEED = 42

    # Breath sequence length (fixed for this dataset)
    SEQ_LEN = 80

    # Feature Definitions
    # Categorical features to be embedded
    CAT_FEATURES = ["R", "C"]

    # Continuous features (Original + Engineered)
    # u_out is binary but treated as continuous for input to NN
    CONT_FEATURES = [
        "time_step",
        "u_in",
        "u_out",
        # Physics-based integrations
        "u_in_cumsum",
        # Interactions
        "R_u_in",  # R * u_in
        "u_in_cumsum_div_C",  # u_in_cumsum / C
        # Dynamics (Lags & Diffs)
        "u_in_lag1",
        "u_in_lag2",
        "u_in_diff1",
        "u_in_diff2",
    ]

    TARGET_COL = "pressure"
    ID_COL = "id"
    BREATH_ID_COL = "breath_id"

    # Dimensions
    NUM_CONT_FEATURES = len(CONT_FEATURES)
    # R has values [5, 20, 50], C has values [10, 20, 50].
    # We will map them to indices 0, 1, 2.
    R_CARDINALITY = 3
    C_CARDINALITY = 3

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Hybrid Architecture Settings
    EMBEDDING_DIM = 8  # Dimension for R and C embeddings
    INPUT_PROJ_DIM = 64  # Project continuous inputs to this dim

    # LSTM Block
    LSTM_HIDDEN_DIM = 512
    LSTM_LAYERS = 4
    LSTM_BIDIRECTIONAL = True

    # Transformer Block (Deprecated per Lesson 00012)
    TRANSFORMER_HEADS = 4
    TRANSFORMER_LAYERS = 2
    TRANSFORMER_FF_DIM = 512

    # General
    DROPOUT = 0.1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    DEBUG = False  # Set to True to run on a small subset for testing
    DEBUG_SAMPLES = 1000  # Number of breaths to use in debug mode

    EPOCHS = 50  # Extended training as per "Idea"
    BATCH_SIZE = 256  # A100 has 40GB, can handle large batches

    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Scheduler settings (Cosine Annealing with Warmup)
    WARMUP_EPOCHS = 5
    ETA_MIN = 1e-5

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # For DataLoader

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print("=" * 30)
        print("CONFIG")
        print("=" * 30)
        for k, v in cls.__dict__.items():
            if not k.startswith("__") and not callable(v):
                print(f"{k}: {v}")
        print("=" * 30)
