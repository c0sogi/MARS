import os
import torch


class Config:
    """
    Configuration module for the Text Normalization task using a
    Confidence-Aware Hybrid Cascade with Curriculum-Enriched Residuals.
    """

    # ==========================================
    # Reproducibility & Debugging
    # ==========================================
    SEED = 42
    DEBUG = False  # Set True to run on a small subset for rapid testing
    DEBUG_SAMPLE_SIZE = 10000  # Number of samples to use in debug mode

    # ==========================================
    # File Paths
    # ==========================================
    # Input Directories (Read-Only)
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Source Data Files
    TRAIN_FILE = os.path.join(METADATA_DIR, "train.csv")
    VAL_FILE = os.path.join(METADATA_DIR, "val.csv")
    TEST_FILE = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "ru_sample_submission_2.csv")

    # Working Directory (Write Access)
    # Using 'idea_8' to isolate this specific experimental run
    WORKING_DIR = "./working/idea_8"

    # Submission Output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Caching & Artifact Paths
    # ==========================================
    # 1. HFBB (Hierarchical Frequency Back-Off) Cache
    HFBB_CACHE_DIR = os.path.join(WORKING_DIR, "hfbb_cache")
    HFBB_UNIGRAM_PATH = os.path.join(HFBB_CACHE_DIR, "unigram.parquet")
    HFBB_BIGRAM_PREV_PATH = os.path.join(HFBB_CACHE_DIR, "bigram_prev.parquet")
    HFBB_BIGRAM_NEXT_PATH = os.path.join(HFBB_CACHE_DIR, "bigram_next.parquet")
    HFBB_TRIGRAM_PATH = os.path.join(HFBB_CACHE_DIR, "trigram.parquet")

    # 2. Transformer Data Cache (Residuals & Anchors)
    DATA_CACHE_DIR = os.path.join(WORKING_DIR, "data_cache")
    # These files contain the specific subset of data for training the neural net
    RESIDUAL_TRAIN_PATH = os.path.join(DATA_CACHE_DIR, "residual_train.parquet")
    RESIDUAL_VAL_PATH = os.path.join(DATA_CACHE_DIR, "residual_val.parquet")

    # 3. Tokenizer Artifacts
    TOKENIZER_DIR = os.path.join(WORKING_DIR, "tokenizers")
    # BPE Model for Target (Normalized Text)
    BPE_MODEL_PREFIX = os.path.join(TOKENIZER_DIR, "bpe_ru_target")
    BPE_MODEL_PATH = f"{BPE_MODEL_PREFIX}.model"
    BPE_VOCAB_PATH = f"{BPE_MODEL_PREFIX}.vocab"
    # Character Vocabulary for Input Source
    CHAR_VOCAB_PATH = os.path.join(TOKENIZER_DIR, "char_vocab.json")

    # 4. Model Checkpoints
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    MODEL_BEST_PATH = os.path.join(CHECKPOINT_DIR, "transformer_best.pth")

    # ==========================================
    # Hyperparameters
    # ==========================================
    # Tier 1: HFBB Logic
    # If Unigram probability for a token > threshold, we trust it and don't use NN.
    # 0.95 implies we only trust "solved" tokens (e.g., unambiguous words).
    HFBB_CONFIDENCE_THRESHOLD = 0.95

    # Data Processing / Curriculum
    N_FOLDS = 5  # Number of folds for Jackknifing to identify residuals
    ANCHOR_RATIO = 0.20  # Fraction of correct semiotic tokens to keep as "anchors"
    MAX_SEQ_LEN = 128  # Max sequence length for Transformer (input & output)
    BPE_VOCAB_SIZE = 8000  # Vocabulary size for output BPE tokenizer

    # Tier 2: Transformer Architecture
    # Sized for A100 (40GB) capabilities
    D_MODEL = 512
    NHEAD = 8
    NUM_LAYERS = 6
    DIM_FEEDFORWARD = 2048
    DROPOUT = 0.1
    LABEL_SMOOTHING = 0.1

    # Training Configuration
    BATCH_SIZE = 256  # Large batch size for stability
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    EPOCHS = 20
    WARMUP_STEPS = 2000
    EARLY_STOPPING_PATIENCE = 4
    NUM_WORKERS = 12  # Use all available vCPUs for data loading

    # ==========================================
    # Utility Methods
    # ==========================================
    @classmethod
    def setup_dirs(cls):
        """
        Creates the necessary directory structure in the working directory.
        Must be called before processing data or training.
        """
        dirs = [
            cls.WORKING_DIR,
            cls.SUBMISSION_DIR,
            cls.HFBB_CACHE_DIR,
            cls.DATA_CACHE_DIR,
            cls.TOKENIZER_DIR,
            cls.CHECKPOINT_DIR,
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    @staticmethod
    def get_device():
        """Returns the PyTorch device (CUDA if available, else CPU)."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
