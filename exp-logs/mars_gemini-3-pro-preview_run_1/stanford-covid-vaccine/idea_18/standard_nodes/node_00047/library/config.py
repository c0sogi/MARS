import os
import torch
import numpy as np
import random


class Config:
    """
    Configuration class for the RNA Degradation Prediction task.
    Implements settings for the 'Interleaved Wide-Stream BiGRU-MLP' architecture (Idea 18).
    """

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    # Metadata directories (Input)
    METADATA_DIR = "./metadata"
    TRAIN_FILE = os.path.join(METADATA_DIR, "train.parquet")
    VAL_FILE = os.path.join(METADATA_DIR, "val.parquet")
    TEST_FILE = os.path.join(METADATA_DIR, "test.parquet")
    SAMPLE_SUBMISSION_FILE = "./input/sample_submission.csv"

    # Working directories (Output)
    # Using 'idea_18' as the experiment identifier
    WORKING_DIR = "./working/idea_18"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    MODEL_DIR = os.path.join(WORKING_DIR, "model")
    PREDS_DIR = os.path.join(WORKING_DIR, "predictions")
    SUBMISSION_DIR = "./submission"

    # Specific output file paths
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Specifications
    # =========================================================================
    SEQ_LEN = 107
    PRED_LEN = 68  # Number of positions scored in training

    # Target Columns
    # Only these 3 are scored and used for training loss
    SCORED_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    # These are required in the submission file but are not scored (filled with 0)
    UNSCORED_TARGETS = ["deg_pH10", "deg_50C"]
    ALL_TARGETS = SCORED_TARGETS + UNSCORED_TARGETS
    NUM_TARGETS = len(SCORED_TARGETS)

    # Token Mappings (Atomic Embeddings)
    # Nucleotides
    NUCLEOTIDE_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
    VOCAB_SIZE_SEQ = len(NUCLEOTIDE_MAP)

    # Predicted Loop Types (bpRNA)
    # S: Stem, M: Multiloop, I: Internal, B: Bulge, H: Hairpin, E: End, X: External
    LOOP_TYPE_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}
    VOCAB_SIZE_LOOP = len(LOOP_TYPE_MAP)

    # =========================================================================
    # Model Architecture (Interleaved Wide-Stream BiGRU-MLP)
    # =========================================================================
    HIDDEN_DIM = 384  # Width of the residual stream
    NUM_LAYERS = 6  # Number of Interleaved Blocks
    MLP_EXPANSION_FACTOR = 4  # Expansion for Pointwise MLP (384 -> 1536 -> 384)
    DROPOUT = 0.1
    USE_DISTANCE_ENCODING = True  # Use Signed Sinusoidal Pairing Distance

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10
    MAX_GRAD_NORM = 1.0

    # Hardware
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Initialize the environment:
        1. Create necessary directories.
        2. Set random seeds for reproducibility.
        """
        # Create directories
        for directory in [
            cls.WORKING_DIR,
            cls.CACHE_DIR,
            cls.MODEL_DIR,
            cls.PREDS_DIR,
            cls.SUBMISSION_DIR,
        ]:
            os.makedirs(directory, exist_ok=True)

        # Set fixed seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
            # Ensure deterministic behavior where possible
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
