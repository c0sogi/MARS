import os
import torch


class Config:
    """
    Configuration class for the TextCNN pipeline.
    Contains paths, hyperparameters, and global constants.
    """

    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_3"
    SUBMISSION_DIR = "./submission"

    # Metadata Input Files
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Raw Data File (referenced by metadata)
    TRAIN_RAW_FILE = "train.csv"
    TEST_RAW_FILE = "test.csv"

    # Output/Cache Files
    # These paths are used to store processed numpy arrays and artifacts
    TRAIN_TOKENS_PATH = os.path.join(WORKING_DIR, "train_tokens.npy")
    TRAIN_LABELS_PATH = os.path.join(WORKING_DIR, "train_labels.npy")

    VAL_TOKENS_PATH = os.path.join(WORKING_DIR, "val_tokens.npy")
    VAL_LABELS_PATH = os.path.join(WORKING_DIR, "val_labels.npy")

    TEST_TOKENS_PATH = os.path.join(WORKING_DIR, "test_tokens.npy")
    TEST_IDS_PATH = os.path.join(WORKING_DIR, "test_ids.npy")

    # Artifacts
    VOCAB_PATH = os.path.join(WORKING_DIR, "vocab.json")
    MLB_PATH = os.path.join(WORKING_DIR, "mlb.joblib")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "textcnn_best.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Processing Hyperparameters
    # ==========================================
    MAX_LEN = 300  # Maximum sequence length (truncating/padding)
    VOCAB_SIZE = 40000  # Maximum size of the vocabulary
    TOP_K_TAGS = 3000  # Number of most frequent tags to predict (Label Space)
    MIN_FREQ = 2  # Minimum frequency for a word to be included in vocab

    # ==========================================
    # Model Architecture Hyperparameters
    # ==========================================
    EMBED_DIM = 300  # Dimension of the word embedding layer
    KERNEL_SIZES = [
        3,
        4,
        5,
    ]  # Sizes of the 1D convolutional kernels (tri-gram, 4-gram, 5-gram)
    NUM_FILTERS = 128  # Number of filters per kernel size
    DROPOUT = 0.5  # Dropout rate for regularization

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 512  # Batch size for training and evaluation
    LEARNING_RATE = 1e-3  # Learning rate for the Adam optimizer
    EPOCHS = 20  # Maximum number of training epochs
    PATIENCE = 3  # Early stopping patience (epochs without improvement)

    # ==========================================
    # System Configuration
    # ==========================================
    NUM_WORKERS = 4  # Number of subprocesses for data loading
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def initialize():
        """
        Creates necessary working and submission directories if they do not exist.
        """
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)


# Execute initialization upon module import
Config.initialize()
