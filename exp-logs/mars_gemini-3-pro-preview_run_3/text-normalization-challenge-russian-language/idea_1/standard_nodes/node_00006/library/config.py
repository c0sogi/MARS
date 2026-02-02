import os
import torch
import random
import numpy as np


class Config:
    # Data Paths
    TRAIN_DATA_PATH = "./metadata/train.csv"
    VAL_DATA_PATH = "./metadata/val.csv"
    TEST_DATA_PATH = "./metadata/test.csv"

    # Output Paths
    WORKING_DIR = "./working/idea_1/"
    CHECKPOINT_PATH = os.path.join(WORKING_DIR, "model_checkpoint.pt")
    VOCAB_PATH = os.path.join(WORKING_DIR, "vocab.npy")
    SUBMISSION_PATH = "./submission/submission.csv"

    # Model Hyperparameters
    embed_dim = 128
    hidden_dim = 512
    n_layers = 1  # Single layer as per vanilla LSTM description
    dropout = 0.1

    # Training Hyperparameters
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 15
    teacher_forcing_ratio = 0.5
    clip_grad = 1.0
    patience = 3  # For early stopping

    # Data Processing
    max_len = 300  # Based on max input length ~230 + margin for expansion

    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, **kwargs):
        """
        Initialize Config with optional overrides.
        """
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        # Ensure working directory exists
        os.makedirs(self.WORKING_DIR, exist_ok=True)
        # Ensure submission directory exists
        os.makedirs(os.path.dirname(self.SUBMISSION_PATH), exist_ok=True)


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
