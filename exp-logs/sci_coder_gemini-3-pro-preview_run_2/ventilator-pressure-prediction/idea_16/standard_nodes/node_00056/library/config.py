import os
import random
import numpy as np
import torch


class Config:
    """
    Configuration class for the High-Fidelity Selective-Injection BiLSTM pipeline.
    Centralizes file paths, hyperparameters, and reproducibility settings.
    """

    def __init__(self, debug: bool = False):
        # ==========================================
        # 1. File Paths & Directories
        # ==========================================
        self.INPUT_DIR = "./input"
        self.METADATA_DIR = "./metadata"
        # Specific working directory for Idea Optimized
        self.WORKING_DIR = "./working/idea_optimized"
        self.SUBMISSION_DIR = "./submission"

        # Raw Data Files
        self.TRAIN_CSV = os.path.join(self.INPUT_DIR, "train.csv")
        self.TEST_CSV = os.path.join(self.INPUT_DIR, "test.csv")
        self.SAMPLE_SUBMISSION = os.path.join(self.INPUT_DIR, "sample_submission.csv")

        # Metadata Files
        self.TRAIN_META = os.path.join(self.METADATA_DIR, "train_metadata.csv")
        self.VAL_META = os.path.join(self.METADATA_DIR, "val_metadata.csv")
        self.TEST_META = os.path.join(self.METADATA_DIR, "test_metadata.csv")

        # Output Files
        self.SUBMISSION_PATH = os.path.join(self.SUBMISSION_DIR, "submission.csv")

        # Cache Paths (for Data Processing)
        self.TRAIN_CACHE = os.path.join(self.WORKING_DIR, "train_processed.parquet")
        self.VAL_CACHE = os.path.join(self.WORKING_DIR, "val_processed.parquet")
        self.TEST_CACHE = os.path.join(self.WORKING_DIR, "test_processed.parquet")
        self.SCALER_CACHE = os.path.join(self.WORKING_DIR, "scaler_params.npy")
        self.MODEL_CHECKPOINT = os.path.join(self.WORKING_DIR, "best_model.pth")

        # Create necessary mutable directories
        os.makedirs(self.WORKING_DIR, exist_ok=True)
        os.makedirs(self.SUBMISSION_DIR, exist_ok=True)

        # ==========================================
        # 2. Model Hyperparameters (HFSI-BiLSTM)
        # ==========================================
        self.HIDDEN_DIM = 512  # Capacity for Deep Recurrent Backbone
        self.BOTTLENECK_DIM = 64  # Dimension for Context Path projection
        self.N_LAYERS = 4  # Number of BiLSTM layers
        self.DROPOUT = 0.1  # Inter-layer dropout
        self.BIDIRECTIONAL = True

        # ==========================================
        # 3. Training Hyperparameters
        # ==========================================
        self.SEED = 42
        self.BATCH_SIZE = 256  # Optimized for A100 GPU
        self.LEARNING_RATE = 1e-3
        self.WEIGHT_DECAY = 1e-6
        self.EPOCHS = 200  # Stretched horizon convergence
        self.T_MAX = 200  # Cosine Annealing horizon
        self.NUM_WORKERS = 12  # Maximize vCPU usage
        self.PIN_MEMORY = True

        # Loss Weights (Focus on Inspiratory Phase)
        self.LOSS_WEIGHT_INSPIRATORY = 1.0
        self.LOSS_WEIGHT_EXPIRATORY = 0.1

        # ==========================================
        # 4. Debug / Development Mode
        # ==========================================
        self.DEBUG = debug
        if self.DEBUG:
            self.EPOCHS = 2
            self.BATCH_SIZE = 64
            self.HIDDEN_DIM = 64
            self.N_LAYERS = 2
            self.NUM_WORKERS = 0  # Disable multiprocessing for debugging

    def seed_everything(self):
        """
        Sets random seeds for Python, NumPy, and PyTorch to ensure reproducibility.
        """
        random.seed(self.SEED)
        os.environ["PYTHONHASHSEED"] = str(self.SEED)
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)
        torch.cuda.manual_seed(self.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def __repr__(self):
        return (
            f"Config(debug={self.DEBUG}, epochs={self.EPOCHS}, "
            f"batch_size={self.BATCH_SIZE}, hidden_dim={self.HIDDEN_DIM}, "
            f"layers={self.N_LAYERS})"
        )
