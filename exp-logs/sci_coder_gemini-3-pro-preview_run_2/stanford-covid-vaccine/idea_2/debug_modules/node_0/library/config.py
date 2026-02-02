import os
import torch


class Config:
    """
    Configuration class for the RNA Degradation Prediction task.
    Centralizes all hyperparameters, file paths, and model specifications
    for the Graph-Enhanced Hybrid Network (GEHN).
    """

    def __init__(self, debug=False):
        """
        Initialize configuration.

        Args:
            debug (bool): If True, sets parameters for a quick debugging run
                          (fewer epochs, subset of data) to validate the pipeline.
        """
        self.debug = debug

        # =============================================================================
        # Reproducibility
        # =============================================================================
        self.seed = 42

        # =============================================================================
        # File Paths
        # =============================================================================
        self.input_dir = "./input"
        self.metadata_dir = "./metadata"
        self.working_dir = "./working/idea_2"

        # Ensure working directory exists
        os.makedirs(self.working_dir, exist_ok=True)

        # Metadata files generated in the previous step
        self.train_file = os.path.join(self.metadata_dir, "train.csv")
        self.val_file = os.path.join(self.metadata_dir, "val.csv")
        self.test_file = os.path.join(self.metadata_dir, "test.csv")
        self.sample_submission = os.path.join(self.input_dir, "sample_submission.csv")

        # Output paths for model artifacts and predictions
        self.best_model_path = os.path.join(self.working_dir, "best_model.pth")
        self.predictions_path = os.path.join(self.working_dir, "predictions.npy")
        self.submission_path = os.path.join(self.working_dir, "submission.csv")

        # Cache directory for processed data (e.g., graphs)
        self.cache_path = os.path.join(self.working_dir, "data_cache")
        os.makedirs(self.cache_path, exist_ok=True)

        # =============================================================================
        # Data Parameters
        # =============================================================================
        self.seq_len = 107
        self.pred_len = 68
        self.num_targets = 5

        # Target columns available in training data
        self.target_cols = [
            "reactivity",
            "deg_Mg_pH10",
            "deg_pH10",
            "deg_Mg_50C",
            "deg_50C",
        ]

        # Columns explicitly scored by the competition metric
        self.scored_cols = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

        # Input Feature Dimensions
        # Sequence (A, G, C, U) -> 4
        # Structure ((, ), .) -> 3
        # Loop Type (S, M, I, B, H, E, X) -> 7
        self.input_channels = 4 + 3 + 7  # Total: 14

        # =============================================================================
        # Model Architecture (GEHN)
        # =============================================================================
        # 1. Dilated CNN Backbone
        # Captures local and medium-range sequence motifs
        self.cnn_channels = 32
        self.cnn_kernel_size = 3
        self.cnn_dilations = [1, 2, 4, 8, 16]  # Exponential dilation factors

        # 2. Graph Neural Network (GNN)
        # Refines features based on secondary structure connectivity
        self.gnn_in_channels = self.cnn_channels
        self.gnn_hidden_channels = 64
        self.gnn_out_channels = 64
        self.gnn_heads = 4  # Number of attention heads (for GAT)
        self.gnn_layers = 2  # Number of GNN layers
        self.gnn_dropout = 0.2

        # 3. BiGRU Head
        # Aggregates global context. Hidden size set to half of input to maintain dimension.
        self.rnn_input_size = self.gnn_out_channels
        self.rnn_hidden_size = self.rnn_input_size // 2
        self.rnn_layers = 1

        # 4. Final Projection
        self.dropout = 0.4  # Dropout before the final linear layer

        # =============================================================================
        # Training Hyperparameters
        # =============================================================================
        self.batch_size = 16
        self.epochs = 50 if not debug else 2
        self.lr = 1e-3
        self.weight_decay = 1e-4
        self.patience = 10  # Early stopping patience
        self.num_workers = 2

        # Subset size for debugging/testing
        self.subset_size = 100 if debug else None

        # =============================================================================
        # Compute
        # =============================================================================
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __str__(self):
        """Pretty print configuration."""
        return str(self.__dict__)
