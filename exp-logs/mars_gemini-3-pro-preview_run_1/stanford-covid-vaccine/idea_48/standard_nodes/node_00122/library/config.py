import os
import torch


class Config:
    """
    Configuration class for the Topologically-Augmented Wide-Stream Residual BiLSTM model.
    Centralizes all hyperparameters, file paths, and execution settings.
    """

    def __init__(self, debug=False, epochs=20, batch_size=16):
        # =========================================================================
        # Paths
        # =========================================================================
        self.input_dir = "./input"
        self.metadata_dir = "./metadata"
        self.working_dir = "./working/idea_48"
        self.submission_dir = "./submission"

        # Create working and submission directories if they don't exist
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)

        # File Paths
        self.train_file = os.path.join(self.metadata_dir, "train.parquet")
        self.val_file = os.path.join(self.metadata_dir, "val.parquet")
        self.test_file = os.path.join(self.metadata_dir, "test.parquet")
        self.sample_submission_file = os.path.join(
            self.input_dir, "sample_submission.csv"
        )
        self.submission_file = os.path.join(self.submission_dir, "submission.csv")

        # =========================================================================
        # Data Processing Parameters
        # =========================================================================
        self.seq_len = 107
        self.pred_len = 68

        # Targets to train on and predict
        self.target_cols = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

        # Random Walk Positional Encoding (RWPE) steps (Powers of Transition Matrix)
        self.rwpe_steps = [1, 2, 4, 8, 16]

        # =========================================================================
        # Model Hyperparameters
        # =========================================================================
        # Topologically-Augmented Wide-Stream Residual BiLSTM
        self.hidden_dim = 512  # Wide stream width
        self.n_layers = 6  # Number of residual blocks
        self.dropout = 0.2  # Inter-layer dropout

        # Embedding Dimensions
        self.seq_embed_dim = 128  # Nucleotide identity
        self.loop_embed_dim = 64  # Predicted loop type
        self.dist_embed_dim = 64  # Sinusoidal pairing distance
        self.rwpe_embed_dim = 32  # Random Walk Structural Fingerprint

        # =========================================================================
        # Training Hyperparameters
        # =========================================================================
        self.batch_size = batch_size
        self.lr = 1e-3
        self.weight_decay = 1e-4  # Low weight decay to preserve recurrent signal
        self.clip_grad = 1.0  # Critical for stabilizing 512-width BiLSTM
        self.epochs = epochs
        self.seed = 42
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # =========================================================================
        # Debug / Execution Flags
        # =========================================================================
        self.debug = debug
        if self.debug:
            self.epochs = 2
            self.batch_size = 4
            # In debug mode, we might want to process fewer samples,
            # handled by the data loader using this flag if needed.
