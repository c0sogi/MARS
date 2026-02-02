import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration class for the Quad-Hybrid Bi-LSTM Tagger with Explicit Features
    and Transformer Fallback.
    """

    def __init__(self, debug=False):
        self.DEBUG = debug
        self.SEED = 42

        # =========================================================================
        # Paths
        # =========================================================================
        self.INPUT_DIR = "./input"
        self.METADATA_DIR = "./metadata"
        self.WORKING_DIR = "./working/idea_10"
        self.SUBMISSION_DIR = "./submission"

        # Ensure working and submission directories exist
        os.makedirs(self.WORKING_DIR, exist_ok=True)
        os.makedirs(self.SUBMISSION_DIR, exist_ok=True)

        # Input Files (Metadata)
        self.TRAIN_FILE = os.path.join(self.METADATA_DIR, "train.csv")
        self.VAL_FILE = os.path.join(self.METADATA_DIR, "val.csv")
        self.TEST_FILE = os.path.join(self.METADATA_DIR, "test.csv")
        self.SUBMISSION_FILE = os.path.join(self.SUBMISSION_DIR, "submission.csv")

        # Output Artifacts
        self.TAGGER_MODEL_PATH = os.path.join(self.WORKING_DIR, "tagger_best_model.pth")
        self.SEQ2SEQ_MODEL_PATH = os.path.join(
            self.WORKING_DIR, "seq2seq_best_model.pth"
        )
        self.KB_PATH = os.path.join(self.WORKING_DIR, "knowledge_base.parquet")

        # Tokenizer / Vocab Artifacts
        self.BPE_MODEL_PREFIX = os.path.join(self.WORKING_DIR, "bpe_tokenizer")
        self.VOCAB_DIR = os.path.join(self.WORKING_DIR, "vocabs")
        os.makedirs(self.VOCAB_DIR, exist_ok=True)

        # Caching Directory
        self.CACHE_DIR = os.path.join(self.WORKING_DIR, "cache")
        os.makedirs(self.CACHE_DIR, exist_ok=True)

        # =========================================================================
        # Data Hyperparameters
        # =========================================================================
        # Max tokens per sentence (EDA shows max is 233, so 300 is safe)
        self.MAX_SENT_LEN = 300
        # Max characters per token (for Char CNN and Seq2Seq)
        self.MAX_TOKEN_CHAR_LEN = 50

        # Vocabulary Sizes
        self.WORD_VOCAB_SIZE = 60000
        self.CHAR_VOCAB_SIZE = 256  # ASCII + Special
        self.BPE_VOCAB_SIZE = 16000

        # Explicit Features
        # Number of regex-based binary features (e.g., is_digit, is_upper, etc.)
        self.NUM_REGEX_FEATURES = 16

        # =========================================================================
        # Model Hyperparameters
        # =========================================================================

        # --- Stage 1: Quad-Hybrid Bi-LSTM Tagger ---
        self.TAGGER_WORD_EMBED_DIM = 128
        self.TAGGER_CHAR_EMBED_DIM = 32
        self.TAGGER_BPE_EMBED_DIM = 64
        self.TAGGER_HIDDEN_DIM = 256
        self.TAGGER_NUM_LAYERS = 2
        self.TAGGER_DROPOUT = 0.3

        # Char CNN Params
        self.CNN_FILTERS = 64
        self.CNN_KERNEL_SIZE = 3

        # --- Stage 2: Transformer Seq2Seq Fallback ---
        self.SEQ2SEQ_EMBED_DIM = 128
        self.SEQ2SEQ_HIDDEN_DIM = 256
        self.SEQ2SEQ_NUM_HEADS = 4
        self.SEQ2SEQ_NUM_LAYERS = 3
        self.SEQ2SEQ_DROPOUT = 0.2

        # =========================================================================
        # Training Hyperparameters
        # =========================================================================
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.NUM_WORKERS = 4

        # Adjust based on Debug mode
        if self.DEBUG:
            self.BATCH_SIZE = 16
            self.NUM_EPOCHS = 2
            self.SAMPLE_SIZE = 5000  # Restrict dataset size for debugging
            self.PATIENCE = 1
        else:
            self.BATCH_SIZE = 128
            self.NUM_EPOCHS = 15
            self.SAMPLE_SIZE = None  # Use full dataset
            self.PATIENCE = 3

        self.LEARNING_RATE = 1e-3
        self.WEIGHT_DECAY = 1e-5
        self.CLIP_GRAD = 1.0

    def set_seed(self):
        """Sets the random seed for reproducibility."""
        random.seed(self.SEED)
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
