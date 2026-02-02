import os
import json
import hashlib
from dataclasses import dataclass, asdict


@dataclass
class Config:
    """
    Configuration class for the Context-Anchored Hybrid Neuro-Symbolic System.
    Centralizes all hyperparameters and handles artifact versioning via hashing.
    """

    # --- Directory Paths ---
    input_dir: str = "./input"
    metadata_dir: str = "./metadata"
    working_dir: str = "./working/idea_5"

    # --- Data Processing Hyperparameters ---
    # Context window size: number of tokens to include before and after the target token
    # Idea specifies +/- 2 words
    context_window: int = 2

    # Maximum sequence length for the character-level neural model inputs
    max_seq_len: int = 128

    # Ratio of PLAIN tokens to sample as background context for the neural model
    # Idea specifies 30-50% to prevent context starvation
    # Cite solution_lesson_node_00012: Do not starve a specialized model of "easy" background data.
    plain_subset_ratio: float = 0.15

    # Random seed for reproducibility across splits and initialization
    seed: int = 42

    # --- Model Architecture (Char-Level Transformer) ---
    d_model: int = 256
    nhead: int = 4
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1

    # --- Training Hyperparameters ---
    # Batch size for A100 GPU
    batch_size: int = 256
    learning_rate: float = 1e-4
    epochs: int = 10
    early_stopping_patience: int = 3
    num_workers: int = 4

    # --- N-Gram / Symbolic Hyperparameters ---
    # Order of the N-gram lookup (3 = Trigram)
    ngram_order: int = 3

    def __post_init__(self):
        """Ensure base working directory exists upon initialization."""
        os.makedirs(self.working_dir, exist_ok=True)

    def get_hash(self) -> str:
        """
        Generates a unique 8-character hash based on the configuration parameters.
        This hash is used to version artifacts (models, tokenizers, stats) to ensure
        consistency between data generation and model training.
        """
        # Convert config to dictionary
        params = asdict(self)

        # Exclude paths and runtime args that do not affect the model/data logic
        # Changing these should not invalidate cached data/models
        exclude_keys = {
            "input_dir",
            "metadata_dir",
            "working_dir",
            "num_workers",
            "batch_size",
        }
        filtered_params = {k: v for k, v in params.items() if k not in exclude_keys}

        # Sort keys to ensure deterministic JSON string representation
        param_str = json.dumps(filtered_params, sort_keys=True)

        # Generate SHA256 hash
        full_hash = hashlib.sha256(param_str.encode("utf-8")).hexdigest()

        # Return first 8 characters
        return full_hash[:8]

    @property
    def version_dir(self) -> str:
        """
        Returns the directory path for the current configuration version.
        Creates the directory if it does not exist.
        """
        v_hash = self.get_hash()
        path = os.path.join(self.working_dir, f"v_{v_hash}")
        os.makedirs(path, exist_ok=True)
        return path

    # --- Artifact Paths (Dynamic based on Hash) ---

    @property
    def tokenizer_path(self) -> str:
        """Path to the character-level tokenizer JSON."""
        return os.path.join(self.version_dir, "char_tokenizer.json")

    @property
    def ngram_stats_path(self) -> str:
        """Path to the N-gram statistics numpy file."""
        return os.path.join(self.version_dir, "ngram_stats.npy")

    @property
    def model_checkpoint_path(self) -> str:
        """Path to the best saved neural model checkpoint."""
        return os.path.join(self.version_dir, "neural_normalizer_best.pt")

    @property
    def train_seq_path(self) -> str:
        """Path to the processed training sequences (Parquet)."""
        return os.path.join(self.version_dir, "train_sequences.parquet")

    @property
    def val_seq_path(self) -> str:
        """Path to the processed validation sequences (Parquet)."""
        return os.path.join(self.version_dir, "val_sequences.parquet")

    @property
    def test_seq_path(self) -> str:
        """Path to the processed test sequences (Parquet)."""
        return os.path.join(self.version_dir, "test_sequences.parquet")

    @property
    def submission_path(self) -> str:
        """Path for the final submission file."""
        # Submission is usually in the main working dir or a specific output dir
        return os.path.join(self.working_dir, "submission.csv")
