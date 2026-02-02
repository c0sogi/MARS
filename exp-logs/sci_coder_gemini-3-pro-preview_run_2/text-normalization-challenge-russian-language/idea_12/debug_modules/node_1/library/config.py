import os
import torch
import random
import numpy as np

# =============================================================================
# GLOBAL PATHS & CONSTANTS
# =============================================================================

# Root Directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_12"
SUBMISSION_DIR = "./submission"

# Input Files (Metadata)
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

# Output/Working Directories
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
TOKENIZER_DIR = os.path.join(WORKING_DIR, "tokenizers")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Semiotic Definition
# Matches any token containing a digit OR a latin character (case-insensitive)
SEMIOTIC_REGEX = r"[\d]|[a-zA-Z]"

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# CONFIGURATION CLASS
# =============================================================================


class ModelConfig:
    """
    Configuration for the Density-Maximized Confidence-Gated Hybrid Cascade.
    """

    def __init__(self, debug=False, seed=42):
        self.debug = debug
        self.seed = seed

        # --- Tier 1: HFBB (Statistical) ---
        # Confidence threshold for the Unigram layer in the backoff hierarchy.
        # If P(mode) > threshold, we accept the HFBB prediction.
        # Otherwise, we route to Tier 2 (Transformer).
        self.confidence_threshold = 0.99

        # --- Tier 2: Transformer (Neural) ---
        # Architecture: Heterogeneous Granularity (Char Encoder -> Subword Decoder)
        self.d_model = 512
        self.nhead = 8
        self.num_encoder_layers = 6
        self.num_decoder_layers = 6
        self.dim_feedforward = 2048
        self.dropout = 0.1
        self.activation = "relu"

        # Sequence Lengths
        # Encoder: Char-level. Context is +/- 2 words.
        # Approx 5 words * 15 chars/word + separators ~ 75-100 chars.
        self.max_enc_len = 128
        # Decoder: Subword-level. Output expansion is rarely > 50 subwords.
        self.max_dec_len = 128

        # Tokenizer Settings
        self.char_vocab_size = 300  # Sufficient for ASCII + Cyrillic + Symbols
        self.bpe_vocab_size = 4000  # Target vocabulary size for BPE

        # Context Window (Number of words before/after to include)
        self.context_window = 2

        # --- Training Hyperparameters ---
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.label_smoothing = 0.1
        self.batch_size = 128
        self.num_workers = 4

        # Training Loop
        if self.debug:
            self.num_epochs = 2
            self.patience = 1
            self.subset_size = 10000  # Restrict data for debugging
        else:
            self.num_epochs = 15
            self.patience = 3
            self.subset_size = None  # Use full dataset

    def __repr__(self):
        return str(self.__dict__)


# =============================================================================
# SETUP UTILITIES
# =============================================================================


def setup_environment(seed=42):
    """
    Creates necessary directories and sets random seeds for reproducibility.
    """
    # 1. Create Directories
    directories = [
        WORKING_DIR,
        CACHE_DIR,
        CHECKPOINT_DIR,
        TOKENIZER_DIR,
        SUBMISSION_DIR,
    ]

    for d in directories:
        os.makedirs(d, exist_ok=True)

    # 2. Set Random Seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in cuDNN (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for hash seeding
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"Environment setup complete. Working directory: {WORKING_DIR}")
    print(f"Device: {DEVICE}")
