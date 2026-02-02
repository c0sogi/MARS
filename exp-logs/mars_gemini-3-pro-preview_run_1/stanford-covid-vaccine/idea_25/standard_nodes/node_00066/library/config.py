import os
import torch
import numpy as np
import random


class Config:
    """
    Configuration for the Dynamic-Depth Wide-Stream Residual BiGRU experiment.
    Defines hyperparameters, file paths, and global settings.
    """

    # --------------------------------------------------------------------------
    # General Settings
    # --------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    NUM_WORKERS = 2  # Number of workers for DataLoader

    # --------------------------------------------------------------------------
    # Directories & Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_25"
    SUBMISSION_DIR = "./submission"

    # Data Files (using pre-generated Parquet metadata)
    TRAIN_FILE = os.path.join(METADATA_DIR, "train.parquet")
    VAL_FILE = os.path.join(METADATA_DIR, "val.parquet")
    TEST_FILE = os.path.join(METADATA_DIR, "test.parquet")
    SAMPLE_SUBMISSION_FILE = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Files
    CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    # Temporary submission file in working directory
    TEMP_SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")
    # Final submission location required by the task
    FINAL_SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Specifications
    # --------------------------------------------------------------------------
    SEQ_LEN = 107
    PRED_LEN = 68

    # Targets to train on and predict (Scored columns only)
    # We explicitly discard deg_pH10 and deg_50C for training as per strategy
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    NUM_TARGETS = len(TARGET_COLS)

    # Vocabulary Mappings
    # Atomic Sequence: A, G, C, U
    TOKEN_TO_ID = {"A": 0, "G": 1, "C": 2, "U": 3}
    ID_TO_TOKEN = {v: k for k, v in TOKEN_TO_ID.items()}
    VOCAB_SIZE_SEQ = 4

    # Predicted Loop Type: S, M, I, B, H, E, X
    LOOP_TYPES = ["S", "M", "I", "B", "H", "E", "X"]
    LOOP_TO_ID = {t: i for i, t in enumerate(LOOP_TYPES)}
    VOCAB_SIZE_LOOP = len(LOOP_TYPES)

    # --------------------------------------------------------------------------
    # Model Architecture: Dynamic-Depth Wide-Stream Residual BiGRU
    # --------------------------------------------------------------------------
    # Backbone Dimensions
    HIDDEN_DIM = 512  # The width of the residual stream (W)
    EMBED_DIM = 128  # Dimension for initial embeddings before projection
    NUM_LAYERS = 12  # Significantly deeper network (8-12 layers)
    DROPOUT = 0.1  # Standard dropout rate

    # Stochastic Depth (LayerDrop) Configuration
    # We linearly decay the survival probability p_l from start to end.
    # Survival = 1.0 means never drop. Survival = 0.5 means 50% chance to drop.
    LAYER_DROP_SURVIVAL_START = 1.0
    LAYER_DROP_SURVIVAL_END = 0.5

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 64
    EPOCHS = 20
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    MAX_GRAD_NORM = 1.0

    # Scheduler Settings
    T_MAX = EPOCHS  # For CosineAnnealingLR

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def set_seed(seed=42):
        """Sets fixed random seeds for reproducibility."""
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @classmethod
    def setup(cls):
        """
        Prepares the environment: creates necessary directories and sets seeds.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        cls.set_seed(cls.SEED)
