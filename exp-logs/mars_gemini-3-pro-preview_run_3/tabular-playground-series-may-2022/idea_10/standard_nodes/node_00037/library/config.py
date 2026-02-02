import os
import torch


class Config:
    """
    Configuration for the Multi-Granularity Embedding Network with SE Blocks.
    Centralizes all hyperparameters, paths, and model settings.
    """

    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # File Paths & Directories
    # ==========================================
    # Input Metadata (Stratified Splits)
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output Directories
    WORKING_DIR = "./working/idea_10"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = "./submission"

    # Output Files
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Parameters
    # ==========================================
    ID_COL = "id"
    TARGET_COL = "target"
    # Columns to exclude from features
    IGNORE_COLS = ["id", "source_path", "target"]

    # Feature Engineering
    # f_27 is a string of length 10
    F27_SEQ_LEN = 10
    # Number of bigrams generated from a sequence of length 10 is 9
    F27_BIGRAM_LEN = 9

    # ==========================================
    # Model Architecture
    # ==========================================
    # Embedding Dimensions
    UNIGRAM_EMBED_DIM = 16
    BIGRAM_EMBED_DIM = 8

    # Funnel MLP Backbone
    HIDDEN_LAYERS = [512, 256, 128]

    # Regularization & Attention
    DROPOUT = 0.2
    USE_SE_BLOCK = True  # Enable Squeeze-and-Excitation

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 1024
    EPOCHS = 30

    # Optimization
    LEARNING_RATE = 1e-3  # Max LR for OneCycleLR
    WEIGHT_DECAY = 1e-5  # Calibrated weight decay

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # ==========================================
    # Debugging / Development
    # ==========================================
    # Set to True to run on a small subset of data for quick pipeline verification
    DEBUG = False
    DEBUG_SAMPLES = 10000

    @classmethod
    def create_dirs(cls):
        """Creates necessary working and submission directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
