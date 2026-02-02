import os


class Config:
    def __init__(self, debug=False, epochs=20, batch_size=64):
        """
        Initializes the configuration.

        Args:
            debug (bool): If True, sets parameters for a quick debug run (fewer epochs, smaller batch).
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
        """
        # General Settings
        self.seed = 42
        self.device = "cuda"
        self.debug = debug

        # Directory Paths
        self.input_dir = "./input"
        self.metadata_dir = "./metadata"
        self.working_dir = "./working/idea_9"

        # Ensure working directory exists
        os.makedirs(self.working_dir, exist_ok=True)

        # Metadata File Paths (Parquet)
        self.train_metadata_path = os.path.join(self.metadata_dir, "train.parquet")
        self.val_metadata_path = os.path.join(self.metadata_dir, "val.parquet")
        self.test_metadata_path = os.path.join(self.metadata_dir, "test.parquet")

        # Cache File Paths (Numpy)
        # Using hash-based naming or fixed names as per requirement.
        # Since the class handles config, fixed names in the specific idea folder are safe.
        self.train_cache_path = os.path.join(self.working_dir, "train_data.npy")
        self.val_cache_path = os.path.join(self.working_dir, "val_data.npy")
        self.test_cache_path = os.path.join(self.working_dir, "test_data.npy")

        # Output Paths
        self.model_save_path = os.path.join(self.working_dir, "best_model.pth")
        self.submission_path = os.path.join(self.working_dir, "submission.csv")

        # Data Dimensions
        self.seq_len = 107
        self.pred_len = 68

        # Feature Mappings
        # Sequence: 4 channels
        self.token2int_seq = {x: i for i, x in enumerate("AGUC")}
        # Structure: 3 channels
        self.token2int_struct = {x: i for i, x in enumerate("().")}
        # Predicted Loop Type: 7 channels
        self.token2int_loop = {x: i for i, x in enumerate("SMIBHEX")}

        self.feature_dim = (
            len(self.token2int_seq)
            + len(self.token2int_struct)
            + len(self.token2int_loop)
        )  # 4+3+7=14

        # Model Architecture: Multi-Scale Inception-GRU
        self.inception_kernels = [1, 3, 5]
        self.stem_channels = 128  # Channels per kernel branch
        # Total input to RNN = stem_channels * len(inception_kernels) = 128 * 3 = 384
        self.hidden_dim = self.stem_channels * len(self.inception_kernels)
        self.num_layers = 3
        self.dropout = 0.4
        self.bidirectional = True
        self.num_targets = 5

        # Training Hyperparameters
        self.epochs = 2 if debug else epochs
        self.batch_size = 16 if debug else batch_size
        self.learning_rate = 1e-3
        self.weight_decay = 1e-4

        # Scheduler (Cosine Annealing)
        self.T_max = self.epochs
        self.eta_min = 1e-5

        # Target Columns
        self.target_cols = [
            "reactivity",
            "deg_Mg_pH10",
            "deg_pH10",
            "deg_Mg_50C",
            "deg_50C",
        ]
        self.scored_cols = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

        # Error Columns (Available in Train/Val but not used for unweighted loss)
        self.error_cols = [
            "reactivity_error",
            "deg_error_Mg_pH10",
            "deg_error_pH10",
            "deg_error_Mg_50C",
            "deg_error_50C",
        ]

    def get_config_info(self):
        """Returns a string summary of the configuration."""
        return (
            f"Config: Debug={self.debug}, Epochs={self.epochs}, Batch={self.batch_size}, "
            f"Model=InceptionGRU(k={self.inception_kernels}, d={self.hidden_dim}, l={self.num_layers})"
        )
