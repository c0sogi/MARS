import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration for Prior-Augmented Multi-Granularity Bi-LSTM with Transformer Fallback.
    """

    # =========================================================================
    # 1. GENERAL SETTINGS
    # =========================================================================
    SEED = 42
    IDEA_NAME = "idea_11"

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # For data loading

    # =========================================================================
    # 2. FILE PATHS
    # =========================================================================
    # Input Metadata (Read-Only)
    METADATA_DIR = "./metadata"
    TRAIN_FILE = os.path.join(METADATA_DIR, "train.csv")
    VAL_FILE = os.path.join(METADATA_DIR, "val.csv")
    TEST_FILE = os.path.join(METADATA_DIR, "test.csv")

    # Working Directory (Read/Write)
    WORK_DIR = os.path.join("./working", IDEA_NAME)

    # Cache Directories
    CACHE_DIR = os.path.join(WORK_DIR, "cache")
    VOCAB_DIR = os.path.join(WORK_DIR, "vocabs")

    # Model Checkpoints
    TAGGER_MODEL_PATH = os.path.join(WORK_DIR, "tagger_best_model.pth")
    SEQ2SEQ_MODEL_PATH = os.path.join(WORK_DIR, "seq2seq_best_model.pth")

    # Artifacts
    KNOWLEDGE_BASE_PATH = os.path.join(WORK_DIR, "knowledge_base.parquet")
    PRIORS_PATH = os.path.join(WORK_DIR, "priors.parquet")
    SUBMISSION_PATH = os.path.join(WORK_DIR, "submission", "submission.csv")

    # Tokenizer Paths
    BPE_MODEL_PREFIX = os.path.join(WORK_DIR, "bpe_tokenizer")

    # =========================================================================
    # 3. DATA HYPERPARAMETERS
    # =========================================================================
    # Vocabulary Limits
    MAX_VOCAB_SIZE_WORD = 100000  # High-level semantics
    MAX_VOCAB_SIZE_BPE = 32000  # Subword semantics

    # Sequence Lengths
    MAX_SEQ_LEN = 128  # Max tokens per sentence for Tagger
    MAX_CHAR_LEN = 20  # Max chars per token for CNN
    MAX_SEQ2SEQ_LEN = 128  # Max chars for generation output

    # =========================================================================
    # 4. MODEL ARCHITECTURE: PENTA-HYBRID TAGGER
    # =========================================================================
    # 1. Word Embeddings
    EMBED_DIM_WORD = 256

    # 2. BPE Embeddings
    EMBED_DIM_BPE = 128

    # 3. Character CNN
    EMBED_DIM_CHAR = 64
    CHAR_CNN_FILTERS = 64
    CHAR_CNN_KERNEL_SIZE = 3

    # 4. Explicit Features (Regex)
    # Dimension determined dynamically, but we set a placeholder config
    USE_EXPLICIT_FEATURES = True

    # 5. Global Prior Features
    # Dimension = Number of Classes (determined dynamically)
    PRIOR_DROPOUT = 0.2  # Feature dropout for priors to force robustness

    # Backbone (Bi-LSTM)
    LSTM_HIDDEN_SIZE = 512
    LSTM_LAYERS = 2
    LSTM_BIDIRECTIONAL = True
    LSTM_DROPOUT = 0.3

    # =========================================================================
    # 5. MODEL ARCHITECTURE: TRANSFORMER SEQ2SEQ FALLBACK
    # =========================================================================
    SEQ2SEQ_EMBED_DIM = 256
    SEQ2SEQ_HIDDEN_DIM = 512
    SEQ2SEQ_LAYERS = 3
    SEQ2SEQ_HEADS = 8
    SEQ2SEQ_DROPOUT = 0.1

    # =========================================================================
    # 6. TRAINING HYPERPARAMETERS
    # =========================================================================
    # General
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    EPOCHS = 15

    # Optimization
    PATIENCE = 3  # Early stopping
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 1

    # Loss
    CLASS_WEIGHT_SMOOTHING_ALPHA = 0.5  # Square-root smoothing (N/Nc)^0.5
    LABEL_SMOOTHING = 0.05

    @classmethod
    def setup(cls):
        """
        Sets up the environment:
        1. Creates necessary directories.
        2. Sets random seeds for reproducibility.
        """
        # 1. Create Directories
        directories = [
            cls.WORK_DIR,
            cls.CACHE_DIR,
            cls.VOCAB_DIR,
            os.path.dirname(cls.SUBMISSION_PATH),
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

        # 2. Set Random Seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.SEED)
            torch.cuda.manual_seed_all(cls.SEED)
            # Ensure deterministic behavior where possible
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        print(f"Configuration setup complete. Working directory: {cls.WORK_DIR}")
        print(f"Device: {cls.DEVICE}")
