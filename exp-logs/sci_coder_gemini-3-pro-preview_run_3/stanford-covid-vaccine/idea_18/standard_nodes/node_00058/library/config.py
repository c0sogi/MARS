import os
import torch
import numpy as np
import random


class Config:
    """
    Configuration class for the RNA Degradation Prediction task.
    Implements settings for the Robust Iterative Structural-Refinement BiGRU (RISR-BiGRU) strategy.
    """

    def __init__(self, debug: bool = False):
        # =============================================================================
        # General Settings
        # =============================================================================
        self.debug = debug
        self.seed = 2024
        self.num_workers = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # =============================================================================
        # Data Paths & Directories
        # =============================================================================
        self.input_dir = "./input"
        self.metadata_dir = "./metadata"
        self.working_dir = "./working/idea_18"
        self.submission_dir = "./submission"

        # Ensure working and submission directories exist
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)

        # Cache Directory for deterministic data processing
        # We use .npy/.npz for caching processed tensors
        self.cache_dir = os.path.join(self.working_dir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        # Metadata Files (Parquet format as generated in metadata step)
        self.train_metadata_path = os.path.join(self.metadata_dir, "train.parquet")
        self.val_metadata_path = os.path.join(self.metadata_dir, "val.parquet")
        self.test_metadata_path = os.path.join(self.metadata_dir, "test.parquet")

        # Submission & Reference Files
        self.sample_submission_path = os.path.join(
            self.input_dir, "sample_submission.csv"
        )
        self.submission_path = os.path.join(self.submission_dir, "submission.csv")

        # Model Checkpoint Path
        self.best_model_path = os.path.join(self.working_dir, "best_model.pth")

        # =============================================================================
        # Dataset Parameters
        # =============================================================================
        self.seq_len = 107
        self.pred_len = 68
        self.num_targets = 5

        # Target Columns in the dataset
        self.target_cols = [
            "reactivity",
            "deg_Mg_pH10",
            "deg_pH10",
            "deg_Mg_50C",
            "deg_50C",
        ]

        # Columns used for the competition metric scoring
        self.scored_targets = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

        # Filtering Strategy:
        # Per RISR strategy, we train on the full dataset (SN_filter=0 included)
        # to maximize signal extraction, despite noise.
        self.use_sn_filter = False

        # =============================================================================
        # Model Architecture (RISR-BiGRU)
        # =============================================================================
        # Input Features:
        # 4 (Sequence: A,G,C,U) + 3 (Structure: .,(,)) + 7 (Loop: S,M,I,B,H,E,X)
        self.input_channels = 14

        # Convolutional Stem
        self.conv_filters = 256
        self.conv_kernel_size = 3

        # Recurrent Backbone
        self.hidden_dim = 384
        self.num_layers = 3  # Number of Iterative Refinement Blocks
        self.dropout = 0.1

        # Windowed Structural Interaction Module
        # Defines the size of the local window around a paired base to gather context from
        self.window_size = 3

        # =============================================================================
        # Training Hyperparameters
        # =============================================================================
        # Batch size adjusted for A100 (40GB)
        self.batch_size = 64 if not self.debug else 4

        # Training duration
        self.epochs = 20 if not self.debug else 2

        # Optimization
        self.lr = 1e-3
        self.weight_decay = 1e-4
        self.max_grad_norm = 1.0  # Gradient clipping to stabilize deep RNN

        # Scheduler (CosineAnnealingLR)
        self.T_max = self.epochs
        self.eta_min = 1e-6

    def get_cache_path(self, name: str) -> str:
        """
        Returns a file path for caching processed data.
        """
        return os.path.join(self.cache_dir, f"{name}.npz")


def setup_reproducibility(seed: int):
    """
    Sets random seeds for Python, NumPy, and PyTorch to ensure reproducible results.
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
