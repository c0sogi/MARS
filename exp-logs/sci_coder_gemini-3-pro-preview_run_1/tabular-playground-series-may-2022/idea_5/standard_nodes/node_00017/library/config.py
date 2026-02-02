import os
import torch


class Config:
    # --------------------------------------------------------------------------
    # General Setup
    # --------------------------------------------------------------------------
    PROJECT_NAME = "idea_5"
    SEED = 42
    # Use CUDA if available, else CPU
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------------------------------
    # Paths
    # --------------------------------------------------------------------------
    # Input Data (Metadata)
    # We use the metadata files generated in the previous step
    METADATA_DIR = "./metadata"
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Directories
    # Working directory for checkpoints and cache
    WORKING_DIR = os.path.join("./working", PROJECT_NAME)
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = "./submission"

    # Model Checkpoints and Submission
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Parameters
    # --------------------------------------------------------------------------
    # Feature Definitions
    # Numerical features: f_00 to f_30, excluding f_27
    NUMERICAL_FEATURES = [f"f_{i:02d}" for i in range(31) if i != 27]
    SEQUENCE_FEATURE = "f_27"
    TARGET_COL = "target"
    ID_COL = "id"

    # Sequence Processing
    # f_27 consists of uppercase letters. We map them to indices 1-26.
    # 0 is reserved for padding/unknown.
    # We set VOCAB_SIZE slightly higher to be safe.
    VOCAB_SIZE = 40
    # f_27 is typically length 10. We set a buffer.
    MAX_SEQ_LEN = 15

    # --------------------------------------------------------------------------
    # Model Architecture
    # --------------------------------------------------------------------------
    # Unified Transformer Settings
    EMBED_DIM = 128
    NUM_HEADS = 4
    NUM_TRANSFORMER_LAYERS = 4
    TRANSFORMER_DROPOUT = 0.1

    # MLP Head Settings
    # High capacity MLP for the fused features
    MLP_HIDDEN_LAYERS = [1024, 512, 256]
    MLP_DROPOUT = 0.1

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    # Batch Size scaled for A100 GPU (40GB VRAM)
    BATCH_SIZE = 2048

    # Epochs set high for OneCycleLR to allow proper annealing
    EPOCHS = 30

    # Optimizer & Scheduler (OneCycleLR)
    LEARNING_RATE = 1e-3  # Max LR
    WEIGHT_DECAY = 1e-2
    PCT_START = 0.3  # 30% of training for warm-up
    DIV_FACTOR = 25.0  # Initial LR = Max LR / 25
    FINAL_DIV_FACTOR = 1e4  # Min LR = Initial LR / 10000

    # Early Stopping
    PATIENCE = 5

    # --------------------------------------------------------------------------
    # Debug / Runtime
    # --------------------------------------------------------------------------
    # Flag to run on a small subset for quick pipeline verification
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 20000

    @classmethod
    def setup(cls):
        """Creates necessary output directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
