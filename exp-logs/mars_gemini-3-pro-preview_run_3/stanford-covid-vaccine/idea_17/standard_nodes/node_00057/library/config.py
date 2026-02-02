import os
import torch


class Config:
    """
    Configuration for the Deep Iterative Structural-Refinement BiGRU (DISR-BiGRU) strategy.
    Centralizes all file paths, model hyperparameters, and training settings.
    """

    def __init__(self, debug: bool = False):
        # ==========================
        # General Settings
        # ==========================
        self.debug = debug
        self.seed = 42
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Use available vCPUs
        self.num_workers = 12

        # ==========================
        # Directories & Paths
        # ==========================
        self.input_dir = "./input"
        self.metadata_dir = "./metadata"
        self.working_dir = "./working/idea_17"

        # Ensure working directory exists
        os.makedirs(self.working_dir, exist_ok=True)

        # Metadata Paths (Parquet files)
        self.train_metadata_path = os.path.join(self.metadata_dir, "train.parquet")
        self.val_metadata_path = os.path.join(self.metadata_dir, "val.parquet")
        self.test_metadata_path = os.path.join(self.metadata_dir, "test.parquet")

        # Sample Submission
        self.sample_submission_path = os.path.join(
            self.input_dir, "sample_submission.csv"
        )

        # Cache Paths (for preprocessed tensors)
        # Using .npy for efficient numpy storage
        self.train_cache_path = os.path.join(self.working_dir, "train_cache.npy")
        self.val_cache_path = os.path.join(self.working_dir, "val_cache.npy")
        self.test_cache_path = os.path.join(self.working_dir, "test_cache.npy")

        # Output Paths
        self.model_save_path = os.path.join(self.working_dir, "best_model.pth")
        self.submission_path = os.path.join(self.working_dir, "submission.csv")

        # ==========================
        # Data Specifications
        # ==========================
        self.seq_len = 107
        self.pred_len = 68
        self.num_targets = 5

        # Target Columns
        self.target_cols = [
            "reactivity",
            "deg_Mg_pH10",
            "deg_pH10",
            "deg_Mg_50C",
            "deg_50C",
        ]

        # Columns used for Leaderboard Scoring
        self.scored_cols = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

        # Feature Configuration (One-Hot Encoding)
        # Sequence: A, G, U, C (4)
        # Structure: (, ), . (3)
        # Loop Type: S, M, I, B, H, E, X (7)
        self.input_channels = 4 + 3 + 7  # Total: 14

        # ==========================
        # Model Hyperparameters
        # ==========================
        # Convolutional Stem
        self.conv_filters = 256
        self.conv_kernel_size = 3

        # Recurrent Backbone (BiGRU)
        self.hidden_dim = 384
        self.num_layers = 3  # Number of BiGRU blocks
        self.dropout = 0.3

        # ==========================
        # Training Hyperparameters
        # ==========================
        # If debug is True, run for fewer epochs on a subset
        self.epochs = 50 if not self.debug else 2
        self.batch_size = 64

        # Optimizer (AdamW)
        self.learning_rate = 1e-3
        self.weight_decay = 1e-4

        # Scheduler (Cosine Annealing)
        self.T_max = self.epochs  # For CosineAnnealingLR
        self.eta_min = 1e-6

        # Stability
        self.clip_grad_norm = 1.0  # Gradient Clipping Max Norm

        # Early Stopping
        self.patience = 10

        # Data Subsetting for Debugging
        self.subset_fraction = 0.05 if self.debug else 1.0
