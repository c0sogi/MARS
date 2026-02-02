import os
import json
import hashlib
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    """
    Configuration class for the Robust Density-Maximized Hybrid Cascade.
    Handles hyperparameters, file paths, and hash-based artifact management.
    """

    def __init__(self, debug=False, epochs=15, batch_size=256):
        # General Settings
        self.seed = 42
        self.debug = debug
        self.idea_name = "idea_13"

        # Hardware
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = 4

        # Paths - Input (Metadata)
        self.metadata_dir = "./metadata"
        self.train_file = os.path.join(self.metadata_dir, "train.csv")
        self.val_file = os.path.join(self.metadata_dir, "val.csv")
        self.test_file = os.path.join(self.metadata_dir, "test.csv")

        # Paths - Output
        self.base_working_dir = os.path.join("./working", self.idea_name)
        self.submission_dir = "./submission"
        self.submission_path = os.path.join(self.submission_dir, "submission.csv")

        # Ensure submission directory exists
        os.makedirs(self.submission_dir, exist_ok=True)

        # Tier 1: HFBB (Statistical Baseline) Params
        self.hfbb_confidence_threshold = 0.99  # Threshold for unigram confidence gating

        # Tier 2: Transformer Architecture
        self.d_model = 256
        self.nhead = 4
        self.num_encoder_layers = 4
        self.num_decoder_layers = 4
        self.dim_feedforward = 1024
        self.dropout = 0.1

        # Data Processing & Tokenization
        self.char_vocab_size = 250  # Estimated size for character-level encoder
        self.bpe_vocab_size = 8000  # Target BPE vocabulary size
        self.max_enc_len = 128  # Max length for encoder (chars)
        self.max_dec_len = 64  # Max length for decoder (subwords)
        self.context_window = 2  # Number of context tokens (anchors)

        # Training Hyperparameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = 3e-4
        self.weight_decay = 1e-4
        self.warmup_steps = 2000
        self.label_smoothing = 0.1
        self.max_grad_norm = 1.0
        self.patience = 3  # Early stopping patience

        # Debugging constraints
        self.debug_sample_size = 50000 if self.debug else None

    def _compute_hash(self):
        """
        Computes a unique hash based on configuration parameters that affect
        data processing and model architecture.
        """
        config_dict = {
            "seed": self.seed,
            "debug": self.debug,
            "hfbb_conf": self.hfbb_confidence_threshold,
            "arch": {
                "d_model": self.d_model,
                "nhead": self.nhead,
                "layers": [self.num_encoder_layers, self.num_decoder_layers],
                "ff": self.dim_feedforward,
            },
            "data": {
                "bpe_vocab": self.bpe_vocab_size,
                "max_lens": [self.max_enc_len, self.max_dec_len],
                "context": self.context_window,
            },
        }
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode("utf-8")).hexdigest()[:10]

    def get_artifact_path(self, filename):
        """
        Returns the full path for an artifact file.
        Creates a unique directory based on the config hash to prevent stale artifacts.

        Args:
            filename (str): Name of the file (e.g., 'model.pth', 'vocab.json')

        Returns:
            str: Full path to the artifact.
        """
        config_hash = self._compute_hash()
        artifact_dir = os.path.join(self.base_working_dir, f"run_{config_hash}")

        # Ensure directory exists
        os.makedirs(artifact_dir, exist_ok=True)

        return os.path.join(artifact_dir, filename)

    def print_summary(self):
        """Prints a summary of the configuration."""
        print(f"=== Configuration Summary ===")
        print(f"Idea: {self.idea_name}")
        print(f"Config Hash: {self._compute_hash()}")
        print(f"Device: {self.device}")
        print(f"Debug Mode: {self.debug}")
        print(f"Batch Size: {self.batch_size}, Epochs: {self.epochs}")
        print(f"Artifact Dir: {os.path.dirname(self.get_artifact_path(''))}")
        print(f"=============================")
