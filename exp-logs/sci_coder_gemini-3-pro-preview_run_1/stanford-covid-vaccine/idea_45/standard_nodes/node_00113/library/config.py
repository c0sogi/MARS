import os
import torch


class Config:
    """
    Configuration for the Noise-Stabilized High-Capacity Wide-Stream BiGRU strategy.
    """

    # =========================================================================
    # Paths & Directories
    # =========================================================================
    EXPERIMENT_ID = "idea_46"
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = os.path.join("./working", EXPERIMENT_ID)

    # Data Files (using Parquet metadata as requested)
    TRAIN_FILE = os.path.join(METADATA_DIR, "train.parquet")
    VAL_FILE = os.path.join(METADATA_DIR, "val.parquet")
    TEST_FILE = os.path.join(METADATA_DIR, "test.parquet")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Outputs
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_SAVE_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # =========================================================================
    # Data Specifications
    # =========================================================================
    SEQ_LEN = 107
    PRED_LEN = 68  # Only the first 68 positions are scored

    # Targets used for training and scoring
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # Targets available in data but excluded from training to reduce noise
    IGNORED_COLS = ["deg_pH10", "deg_50C"]

    # =========================================================================
    # Model Architecture
    # =========================================================================
    # Embedding Dimensions (Proportional)
    EMB_DIM_SEQ = 128  # Atomic Sequence
    EMB_DIM_LOOP = 64  # Predicted Loop Type
    EMB_DIM_PAIR = 64  # Signed Sinusoidal Pairing Distance

    # Backbone Dimensions
    HIDDEN_DIM = 384  # Wide Stream Width (Cite solution_lesson_node_00081)
    NUM_LAYERS = 6  # Depth

    # Regularization
    DROPOUT = 0.1  # Inter-layer Dropout
    NOISE_STD = 0.0  # Disable Noise (Cite solution_lesson_node_00112)

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 64
    EPOCHS = 20
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4  # Low weight decay to preserve recurrent signal
    GRAD_CLIP = 1.0  # Critical for stability with high capacity

    # Scheduler
    T_MAX = EPOCHS  # For Cosine Annealing

    # =========================================================================
    # System & Reproducibility
    # =========================================================================
    SEED = 42
    NUM_WORKERS = 2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debugging / Development
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SAMPLES = 100  # Number of samples to use in debug mode

    @classmethod
    def initialize_workspace(cls):
        """Ensures the working directory exists."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
