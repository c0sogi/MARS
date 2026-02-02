import os
import json
import hashlib
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """Sets the random seed for reproducibility across all libraries."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    """
    Global configuration for the Target-Aware Global-Context Hybrid pipeline.
    Handles file paths, hyperparameters, and artifact versioning via hashing.
    """

    def __init__(self):
        # ==========================================
        # 1. Global Settings
        # ==========================================
        self.seed = 42
        seed_everything(self.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = 4  # Optimized for the 12 vCPU environment

        # ==========================================
        # 2. File Paths
        # ==========================================
        # Input Data (Metadata)
        self.input_dir = "./metadata"
        self.train_file = os.path.join(self.input_dir, "train.csv")
        self.val_file = os.path.join(self.input_dir, "val.csv")
        self.test_file = os.path.join(self.input_dir, "test.csv")

        # Sample submission for format reference
        self.sample_submission_path = "./input/ru_sample_submission_2.csv"

        # Working Directory for this specific idea/experiment
        self.working_dir = "./working/idea_4"
        os.makedirs(self.working_dir, exist_ok=True)

        # Final Submission Directory
        self.submission_dir = "./submission"
        os.makedirs(self.submission_dir, exist_ok=True)
        self.submission_path = os.path.join(self.submission_dir, "submission.csv")

        # ==========================================
        # 3. Data Processing & Tokenization
        # ==========================================
        # Max sequence length: 256 chars covers >99% of sentences based on analysis
        self.max_len = 256

        # Special Tokens for Target-Aware Context
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        # Markers to highlight the target token within the full sentence context
        self.tgt_start_token = "<tgt>"
        self.tgt_end_token = "</tgt>"

        self.special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token,
            self.tgt_start_token,
            self.tgt_end_token,
        ]

        # Hybrid Strategy: Ratio of 'PLAIN'/'PUNCT' tokens to include in neural training
        # to ensure the model learns general language structure/grammar.
        self.plain_inclusion_ratio = 0.30

        # ==========================================
        # 4. Model Hyperparameters (Transformer)
        # ==========================================
        self.embedding_dim = 256
        self.nhead = 4
        self.num_encoder_layers = 4
        self.num_decoder_layers = 4
        self.dim_feedforward = 1024
        self.dropout = 0.1

        # ==========================================
        # 5. Training Hyperparameters
        # ==========================================
        # A100 40GB allows for larger batch sizes
        self.batch_size = 128
        self.learning_rate = 3e-4
        self.weight_decay = 1e-5
        self.num_epochs = 15
        self.early_stopping_patience = 3
        self.label_smoothing = 0.1
        self.clip_grad_norm = 1.0

        # ==========================================
        # 6. Artifact Versioning
        # ==========================================
        self.config_hash = self._generate_hash()

        # Artifact Paths (hashed to prevent stale loading)
        self.tokenizer_path = os.path.join(
            self.working_dir, f"tokenizer_{self.config_hash}.json"
        )
        self.ngram_stats_path = os.path.join(
            self.working_dir, f"ngram_stats_{self.config_hash}.npy"
        )
        self.model_best_path = os.path.join(
            self.working_dir, f"neural_model_{self.config_hash}.pt"
        )

        # Cached Processed Data (Parquet)
        self.train_seq_path = os.path.join(
            self.working_dir, f"train_seq_{self.config_hash}.parquet"
        )
        self.val_seq_path = os.path.join(
            self.working_dir, f"val_seq_{self.config_hash}.parquet"
        )
        self.test_seq_path = os.path.join(
            self.working_dir, f"test_seq_{self.config_hash}.parquet"
        )

    def _generate_hash(self):
        """
        Generates a unique 8-character hash based on configuration parameters
        that impact data processing or model architecture.
        """
        params = {
            "seed": self.seed,
            "max_len": self.max_len,
            "plain_ratio": self.plain_inclusion_ratio,
            "emb_dim": self.embedding_dim,
            "layers_enc": self.num_encoder_layers,
            "layers_dec": self.num_decoder_layers,
            "special_tokens": self.special_tokens,
        }
        # Sort keys to ensure deterministic JSON string
        s = json.dumps(params, sort_keys=True)
        return hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
