import os
import torch
import random
import numpy as np


class Config:
    # --------------------------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------------------------
    SEED = 42

    # --------------------------------------------------------------------------
    # Paths
    # --------------------------------------------------------------------------
    # Input Data (Read-Only)
    INPUT_DIR = "./input"
    TRAIN_METADATA = "./metadata/train.csv"
    VAL_METADATA = "./metadata/val.csv"
    TEST_METADATA = "./metadata/test.csv"

    # Working Directory (Read-Write)
    # Stores cache, vocabulary, and model checkpoints
    WORK_DIR = "./working/idea_3"

    # Specific Artifact Paths
    MODEL_PATH = os.path.join(WORK_DIR, "best_model.pth")
    VOCAB_PATH = os.path.join(WORK_DIR, "vocab.npy")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Parameters
    # --------------------------------------------------------------------------
    VOCAB_SIZE = 50000  # Top K frequent words
    MAX_LEN = 128  # Maximum sequence length (covers >99% of data)
    MIN_FREQ = 2  # Minimum frequency to be included in vocab

    # Special Tokens
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    # Note: The task removes a word, so the input to the model effectively
    # has a "gap". We don't necessarily need a MASK token in the input
    # sequence itself if we are predicting the location of the missing word
    # relative to existing tokens, but we define it just in case.
    MASK_TOKEN = "<MASK>"

    # Debugging / Development
    DEBUG = False  # Set to True to use a small subset of data
    DEBUG_SAMPLE_SIZE = 50000  # Number of samples to use in debug mode

    # --------------------------------------------------------------------------
    # Model Architecture (Decoupled Localization-Classification Transformer)
    # --------------------------------------------------------------------------
    D_MODEL = 512
    N_LAYERS = 6
    N_HEADS = 8
    DIM_FEEDFORWARD = 2048
    DROPOUT = 0.1
    ACTIVATION = "gelu"

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 256  # Optimized for A100 40GB
    LEARNING_RATE = 1e-4  # Base learning rate
    WEIGHT_DECAY = 1e-4  # Regularization
    MAX_EPOCHS = 10  # Maximum training epochs
    PATIENCE = 3  # Early stopping patience

    # Loss Weights for Multi-Task Objective
    # Total Loss = lambda_loc * BCE_Loss + lambda_id * CE_Loss
    LAMBDA_LOC = 1.0  # Weight for Localization (Binary Classification)
    LAMBDA_ID = 1.0  # Weight for Identification (Word Classification)

    # --------------------------------------------------------------------------
    # System / Hardware
    # --------------------------------------------------------------------------
    NUM_WORKERS = 4  # Number of dataloader workers (12 vCPUs available)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PIN_MEMORY = True if torch.cuda.is_available() else False

    @classmethod
    def setup(cls):
        """
        Initialize necessary directories.
        Should be called at the start of execution.
        """
        os.makedirs(cls.WORK_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def set_seed(cls, seed=None):
        """
        Set random seeds for reproducibility.
        """
        if seed is None:
            seed = cls.SEED

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Ensure deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# Automatically setup directories when config is imported
Config.setup()
