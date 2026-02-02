import os
import torch


class Config:
    def __init__(self):
        # ==========================================
        # General Settings
        # ==========================================
        self.seed = 42
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_workers = 2  # Number of workers for data loading

        # Debugging / Development
        self.debug = False
        self.debug_subset_size = 50  # Number of samples to use if debug is True

        # ==========================================
        # File Paths
        # ==========================================
        self.input_dir = "./input"
        self.metadata_dir = "./metadata"
        self.working_dir = "./working/idea_38"
        self.submission_dir = "./submission"

        # Input Data
        self.train_data_path = os.path.join(self.metadata_dir, "train.parquet")
        self.val_data_path = os.path.join(self.metadata_dir, "val.parquet")
        self.test_data_path = os.path.join(self.metadata_dir, "test.parquet")
        self.sample_submission_path = os.path.join(
            self.input_dir, "sample_submission.csv"
        )

        # Outputs
        self.model_save_path = os.path.join(self.working_dir, "best_model.pth")
        self.submission_path = os.path.join(self.submission_dir, "submission.csv")

        # Cache (for preprocessed tensors if needed)
        self.train_cache_path = os.path.join(self.working_dir, "train_cache.npy")
        self.val_cache_path = os.path.join(self.working_dir, "val_cache.npy")
        self.test_cache_path = os.path.join(self.working_dir, "test_cache.npy")

        # Create necessary output directories
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)

        # ==========================================
        # Data Specifications
        # ==========================================
        self.seq_len = 107
        self.pred_len = 68

        # Feature Dimensions:
        # Sequence (4: A,G,U,C) + Structure (3: .,(,)) + LoopType (7: S,M,I,B,H,E,X)
        self.num_features = 14

        self.num_targets = 5
        self.target_cols = [
            "reactivity",
            "deg_Mg_pH10",
            "deg_pH10",
            "deg_Mg_50C",
            "deg_50C",
        ]
        # Only these columns are used for the competition metric
        self.scored_cols = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

        # ==========================================
        # Model Hyperparameters
        # ==========================================
        # Architecture: Deep Decoupled Post-Norm BiGRU
        self.hidden_dim = 384
        self.num_layers = 4

        # Convolutional Stem
        self.cnn_filters = 256
        self.cnn_kernel_size = 3

        # Regularization
        self.dropout = 0.1

        # ==========================================
        # Training Hyperparameters
        # ==========================================
        self.batch_size = 64
        self.learning_rate = 1e-3
        self.weight_decay = 1e-4  # Standard for AdamW
        self.epochs = 25

        # Early Stopping
        self.patience = 5

        # Stability
        self.max_grad_norm = 1.0  # Gradient Clipping (Critical for deep RNNs)

        # Scheduler (Cosine Annealing)
        self.T_max = 25
        self.eta_min = 1e-6
