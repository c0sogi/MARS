import os
import torch


class Config:
    # ==========================================
    # 1. File Paths & Directories
    # ==========================================
    # Input Data
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Working Directory for Idea 2 (NBOW)
    WORKING_DIR = "./working/idea_2"

    # Submission Directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cached Artifacts Paths (for deterministic processing)
    VOCAB_PATH = os.path.join(WORKING_DIR, "vocab.json")
    MLB_PATH = os.path.join(WORKING_DIR, "mlb.joblib")

    # Cached Processed Data (Parquet/Numpy)
    TRAIN_TOKENS_PATH = os.path.join(WORKING_DIR, "train_tokens.npy")
    TRAIN_OFFSETS_PATH = os.path.join(WORKING_DIR, "train_offsets.npy")
    TRAIN_LABELS_PATH = os.path.join(WORKING_DIR, "train_labels.npy")

    VAL_TOKENS_PATH = os.path.join(WORKING_DIR, "val_tokens.npy")
    VAL_OFFSETS_PATH = os.path.join(WORKING_DIR, "val_offsets.npy")
    VAL_LABELS_PATH = os.path.join(WORKING_DIR, "val_labels.npy")

    TEST_TOKENS_PATH = os.path.join(WORKING_DIR, "test_tokens.npy")
    TEST_OFFSETS_PATH = os.path.join(WORKING_DIR, "test_offsets.npy")
    TEST_IDS_PATH = os.path.join(WORKING_DIR, "test_ids.npy")

    # Model Checkpoint
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "nbow_model.pth")

    # ==========================================
    # 2. Data Hyperparameters
    # ==========================================
    VOCAB_SIZE = 100000  # Top N most frequent words
    NUM_TAGS = 5000  # Top K most frequent tags
    MIN_WORD_FREQ = 2  # Minimum frequency to consider a word
    MAX_SEQ_LEN = 300  # Truncate text to this length (tokens)

    # Debugging
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SAMPLE_SIZE = 10000  # Number of samples to use in debug mode

    # ==========================================
    # 3. Model Hyperparameters
    # ==========================================
    EMBED_DIM = 128  # Dimension of word embeddings
    HIDDEN_DIM = 256  # Optional hidden layer dimension (if extended)
    DROPOUT = 0.2  # Dropout probability

    # ==========================================
    # 4. Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 512  # Large batch size for efficient NBOW training
    NUM_EPOCHS = 10  # Number of training epochs
    LEARNING_RATE = 1e-3  # Adam learning rate
    WEIGHT_DECAY = 1e-5  # Regularization
    EARLY_STOPPING_PATIENCE = 3
    NUM_WORKERS = 4  # Data loader workers

    # ==========================================
    # 5. Hardware
    # ==========================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_directories():
    """Creates necessary directories for working files and submissions."""
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    print(f"Directories ensured: {Config.WORKING_DIR}, {Config.SUBMISSION_DIR}")
