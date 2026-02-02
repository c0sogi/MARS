import os
import torch
import numpy as np
import random


class Config:
    """
    Configuration class for the RNA Degradation Prediction task.
    Implements the 'Deep Decoupled Post-Norm BiGRU' strategy settings.
    """

    # ==========================================
    # General Settings
    # ==========================================
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 4  # For data loading

    # ==========================================
    # File Paths
    # ==========================================
    input_dir = "./input"
    metadata_dir = "./metadata"
    working_dir = "./working/idea_37"

    # Metadata files (Pre-split and stratified)
    train_file = os.path.join(metadata_dir, "train.parquet")
    val_file = os.path.join(metadata_dir, "val.parquet")
    test_file = os.path.join(metadata_dir, "test.parquet")

    # Submission template
    sample_submission = os.path.join(input_dir, "sample_submission.csv")

    # Output paths
    model_save_path = os.path.join(working_dir, "best_model.pth")
    submission_save_path = os.path.join(working_dir, "submission.csv")

    # ==========================================
    # Data Dimensions & Features
    # ==========================================
    seq_len = 107
    pred_len = 68  # seq_scored

    # Input Features: 4 (Sequence: A,G,U,C) + 3 (Structure: .,(,)) + 7 (Loop: S,M,I,B,H,E,X)
    num_features = 14

    # Targets
    num_targets = 5
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Scoring: Only these 3 columns are used for the MCRMSE metric
    scored_cols = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    scored_indices = [
        0,
        1,
        3,
    ]  # Indices corresponding to the scored_cols in target_cols

    # ==========================================
    # Model Architecture Hyperparameters
    # ==========================================
    # Convolutional Stem
    conv_filters = 256
    conv_kernel = 3

    # Deep Backbone (BiGRU)
    hidden_dim = 384  # As specified in strategy
    num_layers = 4  # As specified in strategy
    dropout = 0.1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    batch_size = 64
    lr = 1e-3
    weight_decay = 1e-4  # For AdamW
    num_epochs = 50

    # Stability
    gradient_clip = 1.0  # Mandatory for deep hybrid architecture stability

    # Scheduler (Cosine Annealing)
    T_max = 50
    eta_min = 1e-6

    # Early Stopping
    patience = 7

    def __init__(self):
        """
        Initializes the configuration and ensures necessary directories exist.
        """
        os.makedirs(self.working_dir, exist_ok=True)

    @staticmethod
    def set_seed(seed=42):
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
            # Deterministic operations usually come at a performance cost,
            # but are essential for exact reproducibility.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
