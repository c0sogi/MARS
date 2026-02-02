import os
import torch
import random
import numpy as np


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Ensure deterministic behavior on CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class Config:
    """
    Central configuration for the Text Normalization task (Idea 15).
    Stores file paths, hyperparameters, and global settings.
    """

    # --------------------------------------------------------------------------
    # General Settings
    # --------------------------------------------------------------------------
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Number of subprocesses for data loading

    # Debugging: If True, limits the dataset size for faster iteration
    DEBUG = False
    DEBUG_SIZE = 50000  # Number of samples to use when DEBUG is True

    # --------------------------------------------------------------------------
    # Directory Paths
    # --------------------------------------------------------------------------
    # Root directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_15"
    SUBMISSION_DIR = "./submission"

    # Input Data Files (Pre-split by metadata generation script)
    TRAIN_FILE = os.path.join(METADATA_DIR, "train.csv")
    VAL_FILE = os.path.join(METADATA_DIR, "val.csv")
    TEST_FILE = os.path.join(METADATA_DIR, "test.csv")

    # Sample submission for format reference
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "en_sample_submission_2.csv")

    # Output Sub-directories
    VOCAB_DIR = os.path.join(WORKING_DIR, "vocabs")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    LOG_DIR = os.path.join(WORKING_DIR, "logs")

    # --------------------------------------------------------------------------
    # Artifact Paths (Vocabs, Caches, Models)
    # --------------------------------------------------------------------------
    # Vocabulary Files
    WORD_VOCAB_FILE = os.path.join(VOCAB_DIR, "vocab_words.json")
    CHAR_VOCAB_FILE = os.path.join(VOCAB_DIR, "vocab_chars.json")
    CLASS_VOCAB_FILE = os.path.join(VOCAB_DIR, "vocab_classes.json")

    # SentencePiece (BPE) Model
    BPE_MODEL_PREFIX = os.path.join(VOCAB_DIR, "bpe_tokenizer")
    BPE_MODEL_FILE = BPE_MODEL_PREFIX + ".model"
    BPE_VOCAB_FILE = BPE_MODEL_PREFIX + ".vocab"

    # Processed Data Cache (Parquet/NPY for fast loading)
    KNOWLEDGE_BASE_FILE = os.path.join(CACHE_DIR, "knowledge_base.parquet")
    PRIORS_FILE = os.path.join(CACHE_DIR, "priors.parquet")

    # Feature Cache Files
    TRAIN_FEATURES_FILE = os.path.join(CACHE_DIR, "train_features.npy")
    VAL_FEATURES_FILE = os.path.join(CACHE_DIR, "val_features.npy")
    TEST_FEATURES_FILE = os.path.join(CACHE_DIR, "test_features.npy")

    # Model Checkpoints
    TAGGER_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "tagger_best_model.pth")
    SEQ2SEQ_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "seq2seq_best_model.pth")

    # Final Submission
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Processing Hyperparameters
    # --------------------------------------------------------------------------
    # Vocabulary Limits
    MAX_WORD_VOCAB_SIZE = 100000
    BPE_VOCAB_SIZE = 30000

    # Sequence Lengths
    MAX_CHAR_LEN = 20  # Max characters per token (for Char-CNN input)
    MAX_SEQ_LEN = 128  # Max length for Seq2Seq generation output

    # --------------------------------------------------------------------------
    # Model Architecture Hyperparameters
    # --------------------------------------------------------------------------

    # === Stage 1: Gated Multi-Granularity Bi-LSTM Tagger ===
    # Input Representations
    TAGGER_EMBED_DIM_WORD = 128
    TAGGER_EMBED_DIM_BPE = 64
    TAGGER_EMBED_DIM_CHAR = 32

    # Character-Level CNN
    TAGGER_CNN_FILTERS = 64
    TAGGER_CNN_KERNEL_SIZE = 3

    # Explicit Features (Regex Flags)
    # Note: Exact number depends on feature extraction logic, typically ~15-20
    NUM_REGEX_FEATURES = 15

    # Recurrent Backbone
    TAGGER_LSTM_HIDDEN = 256
    TAGGER_LSTM_LAYERS = 2
    TAGGER_DROPOUT = 0.3

    # Gated Fusion Mechanism
    # Dropout applied specifically to the Global Prior vector before fusion
    TAGGER_PRIOR_DROPOUT = 0.2

    # === Stage 2: LSTM Seq2Seq Fallback ===
    SEQ2SEQ_EMBED_DIM = 64
    SEQ2SEQ_HIDDEN_DIM = 256
    SEQ2SEQ_LAYERS = 1
    SEQ2SEQ_DROPOUT = 0.2
    SEQ2SEQ_TEACHER_FORCING_RATIO = 0.5

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 256  # Adjust based on VRAM availability

    # Tagger Optimization
    TAGGER_LR = 1e-3
    TAGGER_EPOCHS = 10
    TAGGER_PATIENCE = 3  # Early stopping patience
    TAGGER_LR_FACTOR = 0.5  # ReduceLROnPlateau factor
    TAGGER_LR_PATIENCE = 1  # ReduceLROnPlateau patience

    # Seq2Seq Optimization
    SEQ2SEQ_LR = 1e-3
    SEQ2SEQ_EPOCHS = 15
    SEQ2SEQ_PATIENCE = 3

    @classmethod
    def setup(cls):
        """
        Creates the necessary directory structure for the project.
        Should be called at the start of the pipeline.
        """
        directories = [
            cls.WORKING_DIR,
            cls.SUBMISSION_DIR,
            cls.VOCAB_DIR,
            cls.CHECKPOINT_DIR,
            cls.CACHE_DIR,
            cls.LOG_DIR,
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        print(f"Configuration initialized. Working directory: {cls.WORKING_DIR}")
