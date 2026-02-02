import os
import torch


class Config:
    """
    Configuration for the Interleaved Gap-Token Transformer model.
    """

    # ==========================================
    # Paths
    # ==========================================
    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output directories
    WORKING_DIR = "./working/idea_4"
    SUBMISSION_DIR = "./submission"

    # File artifacts
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    VOCAB_PATH = os.path.join(WORKING_DIR, "vocab.npy")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Configuration
    # ==========================================
    # Vocabulary
    VOCAB_SIZE = 50000
    MIN_FREQ = 2

    # Special Tokens
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    GAP_TOKEN = "<GAP>"

    # Special Token IDs (Indices)
    PAD_IDX = 0
    UNK_IDX = 1
    GAP_IDX = 2

    # Sequence Length
    # Mean sentence length is ~25 words. Interleaved format is 2*N+1.
    # 128 covers most cases comfortably.
    MAX_SEQ_LEN = 128

    # Debugging / Sampling
    # Set DEBUG = True to use a small subset of data for rapid testing
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 50000  # Number of samples to use in debug mode

    # ==========================================
    # Model Architecture
    # ==========================================
    EMBED_DIM = 256
    HIDDEN_DIM = 512
    NUM_LAYERS = 6
    NUM_HEADS = 8
    DROPOUT = 0.1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 1024  # A100-40GB can handle large batches for this model size
    NUM_EPOCHS = 5  # 30M samples take time; 5 epochs fits within 24h
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    MAX_GRAD_NORM = 1.0

    # Early Stopping
    PATIENCE = 2

    # Loss Weights for Multi-Task Learning
    # Balancing the Localization (Binary) and Identification (Multi-class) heads
    LAMBDA_LOC = 1.0
    LAMBDA_ID = 1.0

    # ==========================================
    # Hardware / System
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 12  # Utilizing available vCPUs
    PIN_MEMORY = True

    @classmethod
    def setup(cls):
        """
        Ensures necessary directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def print_config(cls):
        """
        Prints the current configuration.
        """
        print("=" * 30)
        print("CONFIGURATION")
        print("=" * 30)
        print(f"Device: {cls.DEVICE}")
        print(f"Debug Mode: {cls.DEBUG}")
        print(f"Vocab Size: {cls.VOCAB_SIZE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Max Seq Len: {cls.MAX_SEQ_LEN}")
        print(f"Epochs: {cls.NUM_EPOCHS}")
        print(f"Working Dir: {cls.WORKING_DIR}")
        print("=" * 30)


# Initialize directories on import
Config.setup()
