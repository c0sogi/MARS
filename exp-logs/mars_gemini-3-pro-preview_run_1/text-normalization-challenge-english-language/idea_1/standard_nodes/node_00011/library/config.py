import os
import torch
import random
import numpy as np


class Config:
    # ==========================================
    # Paths
    # ==========================================
    # Input Metadata (Generated in previous step)
    METADATA_DIR = "./metadata"
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Directories
    WORKING_DIR = "./working/idea_1"
    SUBMISSION_DIR = "./submission"

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Artifact Paths (Saved/Loaded during processing)
    VOCAB_PATH = os.path.join(WORKING_DIR, "vocab.npy")
    LABEL_ENCODER_PATH = os.path.join(WORKING_DIR, "label_encoder.npy")
    KNOWLEDGE_BASE_PATH = os.path.join(WORKING_DIR, "knowledge_base.npy")
    MODEL_CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Parameters
    # ==========================================
    # Increased sequence length and vocab size to improve input fidelity (Cite solution_lesson_node_00009)
    MAX_LEN = 300  # Maximum sequence length for LSTM
    VOCAB_SIZE = 120000  # Maximum vocabulary size
    UNK_TOKEN = "<UNK>"
    PAD_TOKEN = "<PAD>"
    PAD_TOKEN_ID = 0
    UNK_TOKEN_ID = 1

    # ==========================================
    # Model Parameters (Bi-LSTM)
    # ==========================================
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 512
    NUM_LAYERS = 2
    DROPOUT = 0.3
    BIDIRECTIONAL = True

    # ==========================================
    # Training Parameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 512  # Large batch size for A100
    LEARNING_RATE = 1e-3
    EPOCHS = 15
    EARLY_STOPPING_PATIENCE = 3
    NUM_WORKERS = 4

    # Class weights handling (Optional flag)
    USE_CLASS_WEIGHTS = True

    # ==========================================
    # Debugging / Development
    # ==========================================
    # Set to a specific integer (e.g., 10000) to limit training data for quick testing
    # Set to None to use the full dataset
    MAX_TRAIN_SAMPLES = None
    DEBUG = False

    # ==========================================
    # Hardware
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
