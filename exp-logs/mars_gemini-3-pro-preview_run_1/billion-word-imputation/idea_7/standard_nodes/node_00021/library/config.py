import os
import torch


class Config:
    # --------------------------------------------------------------------------
    # Directory and File Paths
    # --------------------------------------------------------------------------
    # Input Metadata (Read-Only)
    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Working Directory for Experiment 'idea_7'
    # Used for checkpoints, cached data, and temporary files
    WORKING_DIR = "./working/idea_7"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Model Checkpoint
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Cached Data Paths (Parquet/Numpy)
    # These paths are used by the data processing module to save/load deterministic data
    TRAIN_TOKENS_PATH = os.path.join(WORKING_DIR, "train_tokens.parquet")
    VAL_TOKENS_PATH = os.path.join(WORKING_DIR, "val_tokens.parquet")
    TEST_TOKENS_PATH = os.path.join(WORKING_DIR, "test_tokens.parquet")
    VOCAB_SAVE_PATH = os.path.join(WORKING_DIR, "vocab.npy")
    POS_MAP_SAVE_PATH = os.path.join(WORKING_DIR, "pos_map.npy")

    # Submission Output
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Processing Parameters
    # --------------------------------------------------------------------------
    SEED = 42

    # Vocabulary Settings
    VOCAB_SIZE = 30000  # Limit vocabulary to top K frequent words
    MIN_FREQ = 2  # Minimum frequency to be included in vocab

    # Special Tokens
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    GAP_TOKEN = "<GAP>"  # Token inserted between words to detect missing location
    SOS_TOKEN = "<SOS>"  # Start of Sentence
    EOS_TOKEN = "<EOS>"  # End of Sentence

    # Token Indices (to be populated/verified during vocab build)
    PAD_IDX = 0
    UNK_IDX = 1
    GAP_IDX = 2
    SOS_IDX = 3
    EOS_IDX = 4

    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, GAP_TOKEN, SOS_TOKEN, EOS_TOKEN]

    # Sequence Length
    # Mean sentence length is ~25 words. Interleaving gaps doubles length (2N+1).
    # 256 covers the vast majority of sentences after interleaving.
    MAX_SEQ_LEN = 256

    # --------------------------------------------------------------------------
    # Model Architecture Parameters
    # --------------------------------------------------------------------------
    EMBED_DIM = 256
    HIDDEN_DIM = 1024  # Feed-forward network dimension
    NUM_LAYERS = 6  # Number of Transformer Encoder layers
    NUM_HEADS = 8  # Number of Attention Heads
    DROPOUT = 0.1

    # Auxiliary Task: POS Tagging
    # Universal POS tags count is small (~17). We allocate space for them.
    NUM_POS_TAGS = 20

    # --------------------------------------------------------------------------
    # Training Parameters
    # --------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    BATCH_SIZE = 128  # Suitable for A100 40GB
    NUM_EPOCHS = 10
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-2

    # Multi-Task Loss Weights
    LAMBDA_LOC = 1.0  # Weight for Gap Localization (Binary Classification)
    LAMBDA_ID = 1.0  # Weight for Word Identification (Cross Entropy)
    LAMBDA_SYN = 0.5  # Weight for Syntax/POS Prediction (Auxiliary Cross Entropy)

    # Optimization
    GRAD_CLIP = 1.0
    PATIENCE = 3  # Early stopping patience
    WARMUP_PCT = 0.1  # Percentage of steps for learning rate warmup

    # --------------------------------------------------------------------------
    # Debugging / Development
    # --------------------------------------------------------------------------
    # If True, datasets will be subsampled to DEBUG_SIZE for rapid iteration
    DEBUG = False
    DEBUG_SIZE = 50000
