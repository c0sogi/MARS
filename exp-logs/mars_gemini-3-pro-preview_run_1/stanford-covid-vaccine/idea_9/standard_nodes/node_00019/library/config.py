import os
import torch


class Config:
    """
    Configuration for the RNA Degradation Prediction Task.
    Implements settings for the Multi-Task Distance-Aware Residual BiGRU (Idea 9).
    """

    # --------------------------------------------------------------------------
    # General Setup
    # --------------------------------------------------------------------------
    PROJECT_NAME = "RNA_Degradation_Idea9"
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # Utilization of available vCPUs

    # --------------------------------------------------------------------------
    # File Paths
    # --------------------------------------------------------------------------
    # Input Directories (Read-Only)
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working Directory (Write Access)
    WORKING_DIR = "./working/idea_9"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Data Files (Using Pre-generated Parquet Metadata)
    TRAIN_FILE = os.path.join(METADATA_DIR, "train.parquet")
    VAL_FILE = os.path.join(METADATA_DIR, "val.parquet")
    TEST_FILE = os.path.join(METADATA_DIR, "test.parquet")

    # Submission Reference
    SAMPLE_SUBMISSION_FILE = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Artifacts
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Specifications
    # --------------------------------------------------------------------------
    SEQ_LEN = 107
    SCORED_LEN = 68

    # Nucleotide Vocabulary
    # Standard bases mapped 0-3. Index 4 reserved for [MASK] token.
    VOCAB_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
    MASK_TOKEN_ID = 4
    VOCAB_SIZE = 5

    # Loop Type Vocabulary
    # Structural context mapped 0-6.
    LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}
    LOOP_VOCAB_SIZE = 7

    # Targets
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    NUM_TARGETS = 5

    # --------------------------------------------------------------------------
    # Model Hyperparameters
    # --------------------------------------------------------------------------
    # Embeddings
    EMBED_DIM = 128
    DISTANCE_EMBED_DIM = 64  # Dimension for Sinusoidal Pairing Distance Encoding

    # Backbone: Deep Pre-LayerNorm Residual BiGRU
    HIDDEN_DIM = 256
    NUM_LAYERS = 5
    DROPOUT = 0.1

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 30
    PATIENCE = 10  # Early stopping patience

    # --------------------------------------------------------------------------
    # Multi-Task Learning (Masked Nucleotide Reconstruction)
    # --------------------------------------------------------------------------
    MASK_PROB = 0.15  # Percentage of tokens to mask during training
    LAMBDA_AUX = 0.5  # Weight for the reconstruction CrossEntropy loss

    # --------------------------------------------------------------------------
    # Debugging / Development
    # --------------------------------------------------------------------------
    DEBUG = False
    DEBUG_SUBSET_SIZE = 500  # Number of samples to use if DEBUG is True

    @classmethod
    def create_dirs(cls):
        """Creates the necessary working directories if they don't exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
