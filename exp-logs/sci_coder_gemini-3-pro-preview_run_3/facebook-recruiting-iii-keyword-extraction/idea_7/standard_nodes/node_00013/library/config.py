import os
import torch


class Config:
    # ==========================================
    # System & Environment
    # ==========================================
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 12  # Utilizing available vCPUs

    # ==========================================
    # Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_7"
    SUBMISSION_DIR = "./submission"

    # Ensure output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # File Paths
    # ==========================================
    # Raw Data
    RAW_TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
    RAW_TEST_PATH = os.path.join(INPUT_DIR, "test.csv")

    # Metadata (Splits)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Submission
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Caching & Artifacts
    # ==========================================
    # Preprocessing Artifacts
    TOKENIZER_PATH = os.path.join(WORKING_DIR, "tokenizer.json")
    MLB_PATH = os.path.join(WORKING_DIR, "mlb.joblib")

    # Processed Data Cache (Numpy Arrays)
    TRAIN_TOKENS_PATH = os.path.join(WORKING_DIR, "train_tokens.npy")
    TRAIN_LABELS_PATH = os.path.join(WORKING_DIR, "train_labels.npy")
    VAL_TOKENS_PATH = os.path.join(WORKING_DIR, "val_tokens.npy")
    VAL_LABELS_PATH = os.path.join(WORKING_DIR, "val_labels.npy")
    TEST_TOKENS_PATH = os.path.join(WORKING_DIR, "test_tokens.npy")
    TEST_IDS_PATH = os.path.join(WORKING_DIR, "test_ids.npy")

    # Model Weights
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "model_best.pth")

    # ==========================================
    # Data Processing Hyperparameters
    # ==========================================
    VOCAB_SIZE = 30000  # Subword BPE Vocabulary Size
    MAX_LEN = 512  # Max sequence length (Subwords)
    TOP_K_TAGS = 5000  # Number of most frequent tags to predict (Target Class Count)

    # ==========================================
    # Model Architecture (Subword Dilated Wide & Deep)
    # ==========================================
    EMBED_DIM = 256
    NUM_FILTERS = 128  # Filters per dilation rate
    KERNEL_SIZE = 3  # Fixed kernel size
    DILATION_RATES = [1, 2, 3]  # Parallel dilated convolution rates
    DROPOUT = 0.2

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 128  # A100 can handle large batches
    LEARNING_RATE = 1e-3
    EPOCHS = 10
    PATIENCE = 3  # Early stopping patience

    # ==========================================
    # Debugging
    # ==========================================
    DEBUG = False  # Set to True to run on a subset
    DEBUG_SIZE = 50000  # Number of samples for debugging
