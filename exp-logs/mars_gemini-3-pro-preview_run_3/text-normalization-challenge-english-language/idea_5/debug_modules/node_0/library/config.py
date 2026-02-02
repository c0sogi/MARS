import os
import torch


class Config:
    # ==========================================
    # System & Reproducibility
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Number of subprocesses for data loading

    # ==========================================
    # Directory Structure
    # ==========================================
    # Base Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Specific Experiment Directory
    IDEA_DIR = os.path.join(WORKING_DIR, "idea_5")
    os.makedirs(IDEA_DIR, exist_ok=True)

    # Submission Directory
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Paths
    # ==========================================
    # Metadata (Input)
    TRAIN_META = os.path.join(METADATA_DIR, "train.parquet")
    VAL_META = os.path.join(METADATA_DIR, "val.parquet")
    TEST_META = os.path.join(METADATA_DIR, "test.parquet")

    # Symbolic Statistics Cache (Output of Stage 1)
    STATS_TRIGRAM = os.path.join(IDEA_DIR, "stats_trigram.parquet")
    STATS_BIGRAM_LEFT = os.path.join(IDEA_DIR, "stats_bigram_left.parquet")
    STATS_BIGRAM_RIGHT = os.path.join(IDEA_DIR, "stats_bigram_right.parquet")
    STATS_UNIGRAM = os.path.join(IDEA_DIR, "stats_unigram.parquet")

    # Neural Datasets Cache (Output of Stage 2 Preprocessing)
    PROCESSED_TRAIN = os.path.join(IDEA_DIR, "train_processed.parquet")
    PROCESSED_VAL = os.path.join(IDEA_DIR, "val_processed.parquet")
    PROCESSED_TEST = os.path.join(IDEA_DIR, "test_processed.parquet")

    # Tokenizer Artifacts
    TOKENIZER_PATH = os.path.join(IDEA_DIR, "tokenizer.json")

    # Model Artifacts
    MODEL_CHECKPOINT = os.path.join(IDEA_DIR, "transformer_best.pt")

    # ==========================================
    # Data Processing Parameters
    # ==========================================
    # Context: [prev_token] <SEP> [target_token] <SEP> [next_token]
    CONTEXT_WINDOW = 1

    # Max sequence length for character-level processing
    # Sufficient for "prev (5) + sep + target (10) + sep + next (5)" and output text
    MAX_SEQ_LEN = 128

    # Debugging: Set to True to train on a small subset
    DEBUG = False
    DEBUG_SIZE = 50000

    # ==========================================
    # Vocabulary & Tokens
    # ==========================================
    # Standard Special Tokens
    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    SEP_TOKEN = "<SEP>"
    UNK_TOKEN = "<UNK>"

    # Class Tokens for Conditioning (Chain-of-Thought)
    # These match the classes found in the dataset
    CLASS_TOKENS = [
        "<PLAIN>",
        "<PUNCT>",
        "<DATE>",
        "<LETTERS>",
        "<CARDINAL>",
        "<VERBATIM>",
        "<MEASURE>",
        "<ORDINAL>",
        "<DECIMAL>",
        "<MONEY>",
        "<DIGIT>",
        "<ELECTRONIC>",
        "<TELEPHONE>",
        "<TIME>",
        "<FRACTION>",
        "<ADDRESS>",
    ]

    # ==========================================
    # Model Hyperparameters (Transformer)
    # ==========================================
    # Character-level transformer settings
    D_MODEL = 256
    NHEAD = 4
    NUM_ENCODER_LAYERS = 4
    NUM_DECODER_LAYERS = 4
    DIM_FEEDFORWARD = 1024
    DROPOUT = 0.1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 256
    LEARNING_RATE = 3e-4
    NUM_EPOCHS = 15
    PATIENCE = 3  # Early stopping patience
    GRAD_CLIP = 1.0  # Gradient clipping norm
