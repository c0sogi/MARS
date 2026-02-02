import os
import random
import numpy as np
import torch


class Config:
    """
    Global configuration for the DAAN (Decomposable Attention and Alignment Network) pipeline.
    """

    # -------------------------------------------------------------------------
    # Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_4"
    SUBMISSION_DIR = "./submission"

    # Create writable directories if they don't exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    # Raw Input Files (Read-Only)
    TRAIN_JSONL = os.path.join(INPUT_DIR, "simplified-nq-train.jsonl")
    TEST_JSONL = os.path.join(INPUT_DIR, "simplified-nq-test.jsonl")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files (Pre-generated)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Cache Files (Intermediate Processed Data)
    # Vocabulary mapping
    VOCAB_PATH = os.path.join(WORKING_DIR, "vocab.parquet")
    # Pre-trained or initialized embedding matrix
    EMBEDDING_MATRIX_PATH = os.path.join(WORKING_DIR, "embedding_matrix.npy")

    # Flattened datasets (Question-Candidate pairs with token indices)
    TRAIN_FLATTENED_PATH = os.path.join(WORKING_DIR, "train_flattened.parquet")
    VAL_FLATTENED_PATH = os.path.join(WORKING_DIR, "val_flattened.parquet")
    TEST_FLATTENED_PATH = os.path.join(WORKING_DIR, "test_flattened.parquet")

    # Model Checkpoint
    MODEL_PATH = os.path.join(WORKING_DIR, "daan_model.pth")

    # Final Output
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Hyperparameters
    # -------------------------------------------------------------------------
    # Reproducibility
    SEED = 42

    # Data Processing Limits
    MAX_QUESTION_LEN = 32  # Max tokens for questions
    MAX_CANDIDATE_LEN = 256  # Max tokens for candidate long answers
    VOCAB_SIZE = 30000  # Maximum vocabulary size
    UNK_TOKEN = "<UNK>"
    PAD_TOKEN = "<PAD>"

    # Model Architecture
    EMBEDDING_DIM = 100  # Dimension of word embeddings
    HIDDEN_DIM = 128  # Dimension for internal projection layers
    DROPOUT = 0.2  # Dropout rate for regularization

    # Training Configuration
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 5
    PATIENCE = 2  # Early stopping patience (epochs without improvement)
    NEGATIVE_SAMPLING_RATE = 0.2  # Ratio of negative samples to keep during training

    # Inference Thresholds
    TAU_LONG = 0.4  # Probability threshold to predict a Long Answer
    TAU_SHORT = 0.5  # Confidence threshold (start_prob + end_prob) for Short Answer

    # Debugging
    DEBUG = False  # Set to True to use a small subset of data
    DEBUG_SIZE = 2000  # Number of samples to use in debug mode


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
