import os
import torch


class Config:
    """
    Configuration class for the RNA Degradation Prediction task.
    Implements the 'Stabilized Deep Zero-Masked Channel-Gated BiGRU' strategy parameters.
    """

    def __init__(self, debug=False, num_epochs=None):
        # =============================================================================
        # Paths
        # =============================================================================
        self.input_dir = "./input"
        self.metadata_dir = "./metadata"
        self.working_dir = "./working/idea_29"

        # Ensure working directory exists
        os.makedirs(self.working_dir, exist_ok=True)

        # Input Files (Parquet Metadata)
        self.train_file = os.path.join(self.metadata_dir, "train.parquet")
        self.val_file = os.path.join(self.metadata_dir, "val.parquet")
        self.test_file = os.path.join(self.metadata_dir, "test.parquet")
        self.sample_submission_file = os.path.join(
            self.input_dir, "sample_submission.csv"
        )

        # Cache Files (Numpy format for processed tensors)
        self.train_cache = os.path.join(self.working_dir, "train_data_cache.npy")
        self.val_cache = os.path.join(self.working_dir, "val_data_cache.npy")
        self.test_cache = os.path.join(self.working_dir, "test_data_cache.npy")

        # Output Files
        self.model_path = os.path.join(self.working_dir, "best_model.pth")
        self.submission_path = os.path.join(self.working_dir, "submission.csv")

        # =============================================================================
        # Data Parameters
        # =============================================================================
        self.seq_length = 107
        self.seq_scored = 68

        # Input Channels: 4 (ACGU) + 3 (Structure) + 7 (Loop Type) = 14
        self.input_dim = 14

        # Targets: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
        self.num_targets = 5

        # =============================================================================
        # Model Hyperparameters (Stabilized Deep Zero-Masked Channel-Gated BiGRU)
        # =============================================================================
        self.hidden_dim = 384  # As specified for capacity
        self.num_layers = 4  # Deep backbone
        self.kernel_size = 3  # For Convolutional Stem
        self.dropout = 0.1  # Regularization
        self.bidirectional = True  # BiGRU

        # =============================================================================
        # Training Hyperparameters
        # =============================================================================
        self.seed = 42
        self.batch_size = 32  # Fits comfortably on A100 with this model size
        self.learning_rate = 1e-3  # Initial LR
        self.weight_decay = 1e-4  # AdamW weight decay
        self.max_grad_norm = 1.0  # Gradient Clipping (Mandatory for stability)
        self.patience = 10  # Early stopping patience

        # Epoch control
        if num_epochs is not None:
            self.num_epochs = num_epochs
        else:
            self.num_epochs = 50 if not debug else 2

        # =============================================================================
        # Debugging & Hardware
        # =============================================================================
        self.debug = debug
        # If debug is True, limit dataset size to verify pipeline quickly
        self.debug_subset_size = 100 if debug else None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = 4

    def __repr__(self):
        return (
            f"Config(debug={self.debug}, epochs={self.num_epochs}, "
            f"device={self.device}, hidden_dim={self.hidden_dim}, "
            f"layers={self.num_layers})"
        )
