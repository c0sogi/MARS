import os
import torch


class Config:
    """
    Configuration class for the Transformer-Based Dual-Head Sequence Tagger project.
    Centralizes all file paths, model hyperparameters, and training settings.
    """

    # --------------------------------------------------------------------------
    # Directory Setup & File Paths
    # --------------------------------------------------------------------------
    # Base directories
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_3"
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Input Data Paths (Generated Metadata)
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Output Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    TARGET_VOCAB_PATH = os.path.join(WORKING_DIR, "target_vocab.json")

    # Cache Paths (for deterministic data processing)
    TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_processed.parquet")
    VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_processed.parquet")
    TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_processed.parquet")

    # --------------------------------------------------------------------------
    # Model Architecture
    # --------------------------------------------------------------------------
    MODEL_NAME = "distilroberta-base"
    MAX_SEQ_LEN = 128
    HIDDEN_SIZE = 768  # Hidden size for distilroberta-base
    DROPOUT = 0.1

    # --------------------------------------------------------------------------
    # Task-Specific Hyperparameters
    # --------------------------------------------------------------------------
    # Size of the output vocabulary for the word prediction head (Top K frequent words)
    TARGET_VOCAB_SIZE = 50000

    # Special tokens for the target vocabulary
    UNK_TOKEN = "<UNK>"
    PAD_TOKEN = "<PAD>"

    # Loss Weights: Total Loss = Loss_loc + (LAMBDA_WORD * Loss_word)
    LAMBDA_LOC = 1.0
    LAMBDA_WORD = 1.0

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    SEED = 42
    TRAIN_BATCH_SIZE = 32
    VAL_BATCH_SIZE = 64
    LEARNING_RATE = 3e-5
    WEIGHT_DECAY = 0.01
    NUM_EPOCHS = 5
    WARMUP_STEPS = 1000
    MAX_GRAD_NORM = 1.0

    # Early Stopping parameters
    PATIENCE = 2
    MIN_DELTA = 0.001

    # --------------------------------------------------------------------------
    # Hardware & Runtime
    # --------------------------------------------------------------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4

    # Debugging flag to run on a small subset of data
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 10000
