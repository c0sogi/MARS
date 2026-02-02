import os
import torch
import numpy as np
import random


def set_seed(seed=42):
    """
    Sets fixed random seeds for reproducibility across random, numpy, and torch.
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


class Config:
    """
    Central configuration class for the RNA degradation prediction task.
    Encapsulates file paths, model hyperparameters, and training settings.
    """

    def __init__(self, debug=False):
        # General Settings
        self.seed = 42
        self.debug = debug
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Directory Paths
        self.input_dir = "./input"
        self.metadata_dir = "./metadata"
        self.working_dir = "./working/idea_25"

        # Ensure working directory exists for caching and outputs
        os.makedirs(self.working_dir, exist_ok=True)

        # Data File Paths
        self.train_data_path = os.path.join(self.metadata_dir, "train.parquet")
        self.val_data_path = os.path.join(self.metadata_dir, "val.parquet")
        self.test_data_path = os.path.join(self.metadata_dir, "test.parquet")
        self.sample_submission_path = os.path.join(
            self.input_dir, "sample_submission.csv"
        )

        # Output Paths
        self.model_save_path = os.path.join(self.working_dir, "best_model.pth")
        self.submission_path = os.path.join(self.working_dir, "submission.csv")

        # Dataset Dimensions
        self.seq_len = 107  # Total sequence length
        self.seq_scored = 68  # Number of scored positions (ground truth available)
        self.num_targets = 5  # reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

        # Input Feature Dimensions
        # One-Hot Encoding: Sequence (4) + Structure (3) + LoopType (7)
        # Sequence: A, G, C, U
        # Structure: (, ), .
        # LoopType: S, M, I, B, H, E, X
        self.input_dim = 14

        # Model Architecture Hyperparameters
        # Based on Deep Pre-Norm BiGRU with Zero-Masked Structural Channel-Gating strategy
        self.conv_filters = 256
        self.conv_kernel_size = 3
        self.hidden_dim = 384  # Optimized width for balance
        self.num_layers = 4  # Deeper backbone
        self.dropout = 0.1
        self.use_pre_norm = True  # Stabilize deep network
        self.use_zero_masking = True  # Mask unpaired interactions

        # Training Hyperparameters
        self.batch_size = 32
        self.learning_rate = 1e-3
        self.weight_decay = 1e-2
        self.epochs = 20
        self.gradient_clip_val = 1.0  # Mandatory for stability
        self.patience = 5  # Early stopping patience

        # Scheduler Hyperparameters (Cosine Annealing)
        self.T_max = self.epochs
        self.eta_min = 1e-6

        # Debugging
        self.debug_subset_size = 100  # Number of samples to use in debug mode
