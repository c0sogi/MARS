import os
import random
import numpy as np
import torch


class Config:
    """
    Central configuration class for the Deep Averaging Network (DAN) pipeline.
    Stores file paths, hyperparameters, and device settings.
    """

    def __init__(
        self,
        vocab_size=50000,
        max_seq_len=256,
        embedding_dim=200,
        hidden_dim=256,
        dropout=0.3,
        batch_size=64,
        epochs=20,
        learning_rate=1e-3,
        patience=5,
        seed=42,
    ):

        # --- File Paths ---
        self.METADATA_DIR = "./metadata"
        self.TRAIN_DATA_PATH = os.path.join(self.METADATA_DIR, "train.csv")
        self.VAL_DATA_PATH = os.path.join(self.METADATA_DIR, "val.csv")
        self.TEST_DATA_PATH = os.path.join(self.METADATA_DIR, "test.csv")

        self.SUBMISSION_DIR = "./submission"
        self.SUBMISSION_PATH = os.path.join(self.SUBMISSION_DIR, "submission.csv")

        self.WORKING_DIR = "./working/idea_1"
        self.MODEL_PATH = os.path.join(self.WORKING_DIR, "lstm_model.pth")
        self.VOCAB_PATH = os.path.join(self.WORKING_DIR, "vocab.json")
        self.CACHE_DIR = os.path.join(self.WORKING_DIR, "cache_hybrid")

        # Ensure directories exist
        os.makedirs(self.SUBMISSION_DIR, exist_ok=True)
        os.makedirs(self.WORKING_DIR, exist_ok=True)

        # --- Data Hyperparameters ---
        self.VOCAB_SIZE = vocab_size
        self.MAX_SEQ_LEN = max_seq_len
        self.TOKENIZER_LOWERCASE = True

        # --- Model Hyperparameters ---
        self.EMBEDDING_DIM = embedding_dim
        self.HIDDEN_DIM = 128  # Reduced from 256 (Cite solution_lesson_node_00004)
        self.DROPOUT = 0.5  # Increased from 0.3 (Cite solution_lesson_node_00004)
        self.NUM_CLASSES = 3  # Winner A, Winner B, Tie

        # --- Training Hyperparameters ---
        self.BATCH_SIZE = batch_size
        self.EPOCHS = 8
        self.LEARNING_RATE = learning_rate
        self.PATIENCE = patience
        self.SEED = seed

        # --- Hardware ---
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.NUM_WORKERS = 2


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
