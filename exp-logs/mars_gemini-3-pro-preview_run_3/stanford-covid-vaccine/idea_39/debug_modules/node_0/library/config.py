import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration class for the RNA Degradation Prediction task.
    Centralizes hyperparameters, file paths, and model settings.
    """

    def __init__(self, debug=False, epochs=50, batch_size=64, max_train_samples=None):
        """
        Initialize configuration with flexible overrides.

        Args:
            debug (bool): If True, runs in debug mode (e.g., less data).
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training and inference.
            max_train_samples (int, optional): Limit training data size for debugging.
        """
        # General Settings
        self.seed = 42
        self.debug = debug
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = 12  # Utilizing available vCPUs

        # Paths
        self.input_dir = "./input"
        self.metadata_dir = "./metadata"
        self.working_dir = "./working/idea_39"

        # Metadata Paths (Parquet files)
        self.train_metadata_path = os.path.join(self.metadata_dir, "train.parquet")
        self.val_metadata_path = os.path.join(self.metadata_dir, "val.parquet")
        self.test_metadata_path = os.path.join(self.metadata_dir, "test.parquet")

        # Submission Paths
        self.sample_submission_path = os.path.join(
            self.input_dir, "sample_submission.csv"
        )
        self.submission_path = os.path.join(self.working_dir, "submission.csv")
        self.model_save_path = os.path.join(self.working_dir, "best_model.pth")

        # Cache Directory (for processed tensors)
        self.cache_dir = self.working_dir

        # Ensure working directory exists
        os.makedirs(self.working_dir, exist_ok=True)

        # Data Dimensions
        self.seq_len = 107
        self.pred_len = 68
        self.max_train_samples = max_train_samples

        # Feature Mappings (One-Hot Encoding)
        # Sequence: A, G, U, C
        self.token2int = {x: i for i, x in enumerate("AGUC")}
        # Structure: ., (, )
        self.struct2int = {x: i for i, x in enumerate(".()")}
        # Loop Type: S, M, I, B, H, E, X
        self.loop2int = {x: i for i, x in enumerate("SMIBHEX")}

        # Input Channel Dimensions
        self.num_tokens = 4
        self.num_struct = 3
        self.num_loop = 7
        # Total Input Channels = 4 (Seq) + 3 (Struct) + 7 (Loop) = 14
        self.input_dim = self.num_tokens + self.num_struct + self.num_loop

        # Model Architecture (Deep Decoupled Post-Norm BiGRU)
        self.hidden_dim = 384
        self.num_layers = 4
        self.dropout = 0.1
        self.num_classes = 5

        # Training Hyperparameters
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = 1e-3
        self.weight_decay = 1e-4
        self.grad_clip = 1.0  # Mandatory for stability
        self.patience = 10  # Early stopping patience

        # Target Columns
        self.target_cols = [
            "reactivity",
            "deg_Mg_pH10",
            "deg_pH10",
            "deg_Mg_50C",
            "deg_50C",
        ]
        # Columns used for the competition metric
        self.scored_cols = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
