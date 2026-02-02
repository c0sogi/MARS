import os
import torch

# =============================================================================
# GENERAL SETTINGS
# =============================================================================
SEED = 42
DEBUG = False  # Set to True to limit dataset size for debugging
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 12  # Number of dataloader workers (matching available vCPUs)

# =============================================================================
# DIRECTORY PATHS
# =============================================================================
# Input Metadata (Read-Only)
INPUT_DIR = "./metadata"
TRAIN_FILE = os.path.join(INPUT_DIR, "train.csv")
VAL_FILE = os.path.join(INPUT_DIR, "val.csv")
TEST_FILE = os.path.join(INPUT_DIR, "test.csv")

# Working Directory (Write Access) - Idea 10 specific
WORKING_DIR = "./working/idea_10"
os.makedirs(WORKING_DIR, exist_ok=True)

# Sub-directories for Caching and Artifacts
HFBB_CACHE_DIR = os.path.join(WORKING_DIR, "hfbb_cache")
TRANSFORMER_CACHE_DIR = os.path.join(WORKING_DIR, "transformer_cache")
TOKENIZER_DIR = os.path.join(WORKING_DIR, "tokenizers")
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
SUBMISSION_DIR = "./submission"

# Ensure all directories exist
for d in [
    HFBB_CACHE_DIR,
    TRANSFORMER_CACHE_DIR,
    TOKENIZER_DIR,
    CHECKPOINT_DIR,
    SUBMISSION_DIR,
]:
    os.makedirs(d, exist_ok=True)

# =============================================================================
# FILE PATHS
# =============================================================================
# HFBB (Hierarchical Frequency-Based Backoff) Cache Files
UNIGRAM_PATH = os.path.join(HFBB_CACHE_DIR, "unigram.parquet")
BIGRAM_PREV_PATH = os.path.join(HFBB_CACHE_DIR, "bigram_prev.parquet")
BIGRAM_NEXT_PATH = os.path.join(HFBB_CACHE_DIR, "bigram_next.parquet")
TRIGRAM_PATH = os.path.join(HFBB_CACHE_DIR, "trigram.parquet")

# Transformer Processed Data Cache
PROCESSED_TRAIN_PATH = os.path.join(TRANSFORMER_CACHE_DIR, "processed_train.parquet")
PROCESSED_VAL_PATH = os.path.join(TRANSFORMER_CACHE_DIR, "processed_val.parquet")

# Tokenizer Artifacts
CHAR_VOCAB_PATH = os.path.join(TOKENIZER_DIR, "char_vocab.json")
BPE_MODEL_PREFIX = os.path.join(TOKENIZER_DIR, "bpe_ru_target")
BPE_MODEL_PATH = f"{BPE_MODEL_PREFIX}.model"
BPE_VOCAB_PATH = f"{BPE_MODEL_PREFIX}.vocab"

# Model Checkpoints and Submission
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "transformer_best.pth")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# DATA PROCESSING CONFIGURATION
# =============================================================================
# Semiotic classes to be explicitly handled by the Transformer (Tier 2)
# These classes often have complex internal structures or low confidence in HFBB.
SEMIOTIC_CLASSES = [
    "CARDINAL",
    "DATE",
    "LETTERS",
    "VERBATIM",
    "ORDINAL",
    "MEASURE",
    "TELEPHONE",
    "DECIMAL",
    "MONEY",
    "ELECTRONIC",
    "DIGIT",
    "TIME",
    "FRACTION",
    "ADDRESS",
]

# Classes excluded from Transformer training (handled by HFBB or Identity)
EXCLUDE_CLASSES = ["PLAIN", "PUNCT"]

# Class Balancing: Target sample count for upsampling rare classes.
# Rare semiotic classes will be upsampled to match this frequency.
TARGET_CLASS_COUNT = 50000

# Text Processing Limits
MAX_INPUT_LEN = 64  # Maximum character length for input context window
MAX_OUTPUT_LEN = 64  # Maximum subword length for output sequence
VOCAB_SIZE = 4000  # Target BPE vocabulary size (sufficient for normalized forms)

# =============================================================================
# MODEL ARCHITECTURE (TRANSFORMER)
# =============================================================================
# Encoder-Decoder Architecture parameters
D_MODEL = 256
NHEAD = 4
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
DIM_FEEDFORWARD = 1024
DROPOUT = 0.1

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
BATCH_SIZE = 128
LEARNING_RATE = 5e-4
NUM_EPOCHS = 15
WARMUP_STEPS = 1000
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.1
EARLY_STOPPING_PATIENCE = 3
GRAD_CLIP = 1.0

# Debug / Data Control
# If DEBUG is True, limit training data to MAX_TRAIN_SAMPLES for rapid iteration
MAX_TRAIN_SAMPLES = 10000 if DEBUG else None

# =============================================================================
# HYBRID PIPELINE LOGIC
# =============================================================================
# Confidence threshold for HFBB Unigram model.
# Logic:
# 1. Trigram/Bigram Match -> Accept
# 2. Unigram Match AND Confidence > THRESHOLD -> Accept
# 3. Else -> Route to Transformer
CONFIDENCE_THRESHOLD = 0.95
