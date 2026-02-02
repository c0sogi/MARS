import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration class for the RNA Degradation Prediction task.
    Implements the hyperparameters and settings for the
    'Internally-Normalized Wide-Stream Residual BiLSTM' strategy.
    """

    # ==============================
    # General Settings
    # ==============================
    PROJECT_NAME = "RNA_Degradation_Idea_53"
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2  # Adjust based on available vCPUs (12 available)

    # ==============================
    # Data Paths
    # ==============================
    # Root directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_53"  # Specific cache directory requirement
    SUBMISSION_DIR = "./submission"

    # Specific file paths
    TRAIN_FILE = os.path.join(METADATA_DIR, "train.parquet")
    VAL_FILE = os.path.join(METADATA_DIR, "val.parquet")
    TEST_FILE = os.path.join(METADATA_DIR, "test.parquet")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output paths
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # ==============================
    # Data Dimensions & Targets
    # ==============================
    SEQ_LEN = 107
    PRED_LEN = 68

    # We strictly train on these 3 scored columns
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    NUM_CLASSES = len(TARGET_COLS)

    # ==============================
    # Model Architecture
    # ==============================
    # "Internally-Normalized Wide-Stream Residual BiLSTM"

    # Embeddings
    EMBEDDING_DIM = 128  # Atomic Sequence embedding
    LOOP_EMBEDDING_DIM = 64  # Predicted Loop Type embedding
    PAIR_EMBEDDING_DIM = 64  # Signed Sinusoidal Pairing Distance embedding

    # Backbone
    HIDDEN_DIM = 512  # Explicit Width 512
    NUM_LAYERS = 6  # 6 Residual Blocks
    DROPOUT = 0.2  # Inter-layer dropout

    # Note: Internal LayerNorm logic is handled in the model definition,
    # but the dimension settings here support it.

    # ==============================
    # Training Hyperparameters
    # ==============================
    BATCH_SIZE = 32  # Strictly 32
    EPOCHS = 20  # Fixed budget
    LEARNING_RATE = 1e-3  # Standard AdamW start
    WEIGHT_DECAY = 1e-4  # Low weight decay
    CLIP_GRAD = 1.0  # Gradient clipping to norm 1.0 for stability

    # Scheduler
    T_MAX = 20  # Matches EPOCHS for Cosine Annealing

    def __init__(self):
        # Ensure working and submission directories exist
        os.makedirs(self.WORKING_DIR, exist_ok=True)
        os.makedirs(self.SUBMISSION_DIR, exist_ok=True)

    @staticmethod
    def setup_reproducibility(seed=42):
        """
        Sets the seed for random number generators to ensure reproducibility.
        """
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Deterministic algorithms can be slower, but ensure reproducibility
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
