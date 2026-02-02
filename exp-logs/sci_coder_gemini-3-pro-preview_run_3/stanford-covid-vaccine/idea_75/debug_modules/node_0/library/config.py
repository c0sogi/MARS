import os
import torch


class Config:
    """
    Configuration class for the RNA Degradation Prediction task.
    Centralizes hyperparameters, file paths, and model settings.
    """

    def __init__(self, debug: bool = False):
        """
        Initialize configuration.

        Args:
            debug (bool): If True, sets epochs to a low number for quick testing.
        """
        # Reproducibility
        self.seed = 42

        # Compute
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = (
            12 if hasattr(os, "sched_getaffinity") else 4
        )  # Utilize available vCPUs

        # Directory Structure
        self.input_dir = "./input"
        self.metadata_dir = "./metadata"
        self.working_dir = "./working/idea_75"
        self.cache_dir = os.path.join(self.working_dir, "cache")

        # Ensure working and cache directories exist
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        # Input Data Paths (Metadata)
        self.train_file = os.path.join(self.metadata_dir, "train.parquet")
        self.val_file = os.path.join(self.metadata_dir, "val.parquet")
        self.test_file = os.path.join(self.metadata_dir, "test.parquet")
        self.sample_submission_file = os.path.join(
            self.input_dir, "sample_submission.csv"
        )

        # Cache File Paths (for deterministic data processing)
        self.train_cache_path = os.path.join(self.cache_dir, "train_data.npz")
        self.val_cache_path = os.path.join(self.cache_dir, "val_data.npz")
        self.test_cache_path = os.path.join(self.cache_dir, "test_data.npz")

        # Output Paths
        self.model_save_path = os.path.join(self.working_dir, "best_model.pth")
        self.submission_path = os.path.join(self.working_dir, "submission.csv")

        # Data Specifications
        self.seq_len = 107
        self.pred_len = 68
        self.num_targets = 5

        # Input Feature Vocabularies
        self.vocab_size_seq = 4  # A, G, C, U
        self.vocab_size_struct = 3  # ., (, )
        self.vocab_size_loop = 7  # S, M, I, B, H, E, X

        # Model Architecture: High-Capacity Full-Rank GLU-Decoupled BiGRU
        self.hidden_dim = 384  # Hidden dimension per direction (Total 768)
        self.n_layers = 4  # Number of BiGRU + Interaction blocks
        self.kernel_size = 3  # Convolutional stem kernel size
        self.dropout = 0.1  # Dropout rate

        # Training Hyperparameters
        self.batch_size = 32
        self.lr = 1e-3
        self.weight_decay = 1e-4
        self.epochs = 50 if not debug else 2
        self.patience = 7  # Early stopping patience
        self.grad_clip = 1.0  # Gradient clipping max norm (Crucial for stability)
