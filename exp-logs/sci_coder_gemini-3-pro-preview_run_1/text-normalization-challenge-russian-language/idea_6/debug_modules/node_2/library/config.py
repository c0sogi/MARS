import os
import torch
import hashlib
import json
from dataclasses import dataclass, asdict

# --- Global Constants ---
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Special Tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
SEP_TOKEN = "<SEP>"  # Used to separate context from target in the neural input


# --- Configuration Class ---
@dataclass
class Config:
    # --- Paths ---
    # Input Metadata (Read-Only)
    train_data_path: str = "./metadata/train.csv"
    val_data_path: str = "./metadata/val.csv"
    test_data_path: str = "./metadata/test.csv"

    # Working Directory (Read/Write)
    working_dir: str = "./working/idea_6"

    # --- Data Processing ---
    context_window: int = 2  # +/- 2 words
    bpe_vocab_size: int = 32000
    # Character vocab size is determined dynamically but we set a max limit for safety
    max_char_vocab_size: int = 500
    max_seq_len: int = 128  # Max length for the fused sequence

    # --- Model Hyperparameters ---
    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1

    # --- Training Hyperparameters ---
    batch_size: int = 256  # A100 allows for larger batch sizes
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    label_smoothing: float = 0.1
    epochs: int = 15
    early_stopping_patience: int = 3
    grad_clip: float = 1.0

    # --- Execution Control ---
    debug: bool = False  # If True, runs on a small subset
    num_workers: int = 4
    seed: int = SEED

    def __post_init__(self):
        """Ensure working directory exists."""
        os.makedirs(self.working_dir, exist_ok=True)

    def get_run_hash(self) -> str:
        """
        Generates a unique short hash based on the configuration critical to training.
        This allows versioning of artifacts (tokenizers, models, stats) to prevent
        caching collisions when parameters change.
        """
        # Select keys that affect the model architecture and data processing
        # We exclude paths or execution flags (like debug/workers) that don't change the artifact content
        config_dict = {
            "context_window": self.context_window,
            "bpe_vocab_size": self.bpe_vocab_size,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": (self.num_encoder_layers, self.num_decoder_layers),
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "label_smoothing": self.label_smoothing,
            "seed": self.seed,
        }

        # Serialize and hash
        config_str = json.dumps(config_dict, sort_keys=True)
        hash_object = hashlib.md5(config_str.encode("utf-8"))
        return hash_object.hexdigest()[:8]  # Return first 8 chars

    # --- Artifact Path Generators ---
    # These methods use the hash to create versioned filenames

    @property
    def bpe_tokenizer_path(self) -> str:
        return os.path.join(
            self.working_dir, f"bpe_tokenizer_{self.get_run_hash()}.json"
        )

    @property
    def char_tokenizer_path(self) -> str:
        return os.path.join(
            self.working_dir, f"char_tokenizer_{self.get_run_hash()}.json"
        )

    @property
    def ngram_stats_path(self) -> str:
        return os.path.join(self.working_dir, f"ngram_stats_{self.get_run_hash()}.npy")

    @property
    def model_checkpoint_path(self) -> str:
        return os.path.join(self.working_dir, f"neural_model_{self.get_run_hash()}.pt")

    @property
    def train_seq_path(self) -> str:
        return os.path.join(
            self.working_dir, f"train_sequences_{self.get_run_hash()}.parquet"
        )

    @property
    def val_seq_path(self) -> str:
        return os.path.join(
            self.working_dir, f"val_sequences_{self.get_run_hash()}.parquet"
        )

    @property
    def test_seq_path(self) -> str:
        return os.path.join(
            self.working_dir, f"test_sequences_{self.get_run_hash()}.parquet"
        )

    @property
    def submission_path(self) -> str:
        # Submission doesn't need a hash as it's the final output
        return os.path.join(self.working_dir, "submission.csv")
