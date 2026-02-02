import os
import torch


class Config:
    """
    Configuration class for the RNA Degradation Prediction task.
    Implements the 'High-Capacity Residual-Structural Synthesis' strategy settings.
    """

    def __init__(self, debug=False):
        # ==========================================
        # General Settings
        # ==========================================
        self.debug = debug
        self.seed = 42
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ==========================================
        # Paths
        # ==========================================
        self.input_dir = "./input"
        self.metadata_dir = "./metadata"
        self.working_dir = "./working/idea_80"

        # Ensure working directory exists
        os.makedirs(self.working_dir, exist_ok=True)

        # Metadata File Paths (Parquet format)
        self.train_file = os.path.join(self.metadata_dir, "train.parquet")
        self.val_file = os.path.join(self.metadata_dir, "val.parquet")
        self.test_file = os.path.join(self.metadata_dir, "test.parquet")

        # Submission and Output Paths
        self.sample_submission_file = os.path.join(
            self.input_dir, "sample_submission.csv"
        )
        self.submission_file = os.path.join(self.working_dir, "submission.csv")
        self.model_save_path = os.path.join(self.working_dir, "best_model.pth")

        # Cache Paths (for deterministic data processing)
        self.train_cache = os.path.join(self.working_dir, "train_cache.npy")
        self.val_cache = os.path.join(self.working_dir, "val_cache.npy")
        self.test_cache = os.path.join(self.working_dir, "test_cache.npy")

        # ==========================================
        # Model Hyperparameters
        # Strategy: Deep Residual High-Capacity BiGRU with Unified GLU-Refinement
        # ==========================================
        # Input: 4 bases (A,G,C,U) + 3 structure ((,),.) + 7 loop types
        self.input_dim = 14

        # Backbone Configuration
        self.hidden_dim = 384  # Dimension per direction (Total hidden size = 768)
        self.num_layers = 4  # Deep Backbone for maximum capacity
        self.dropout = 0.1  # Conservative Regularization (avoid 0.5)

        # Convolutional Stem Configuration
        self.kernel_size = 3
        self.cnn_filters = 256

        # ==========================================
        # Data Dimensions & Targets
        # ==========================================
        self.seq_len = 107
        self.pred_len = 68
        self.num_targets = 5

        # All 5 targets are predicted, but only 3 are scored
        self.target_cols = [
            "reactivity",
            "deg_Mg_pH10",
            "deg_pH10",
            "deg_Mg_50C",
            "deg_50C",
        ]
        self.scored_cols = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

        # ==========================================
        # Training Hyperparameters
        # ==========================================
        self.batch_size = 16  # Adjusted for VRAM usage with deep model
        self.num_epochs = 50
        self.learning_rate = 1e-3
        self.weight_decay = 1e-4
        self.clip_grad_norm = 1.0  # Mandatory for stability with deep RNNs
        self.patience = 7  # Early stopping patience
        self.num_workers = 4

        # Debugging constraints
        self.max_train_samples = 100 if self.debug else None
        self.max_val_samples = 50 if self.debug else None
