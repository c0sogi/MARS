import os
import torch


class Config:
    """
    Central configuration for the Global-Localization Interleaved Transformer project.
    Contains file paths, model hyperparameters, training settings, and system configurations.
    """

    # ==========================================
    # Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Working directory for Idea 5 (Global-Localization Interleaved Transformer)
    WORKING_DIR = "./working/idea_5"

    # Input Metadata Files (Generated previously)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Cache Files (for deterministic data processing)
    # Using .npy for vocab and .parquet for tokenized datasets
    VOCAB_PATH = os.path.join(WORKING_DIR, "vocab.npy")
    TRAIN_TOKENS_PATH = os.path.join(WORKING_DIR, "train_tokens.parquet")
    VAL_TOKENS_PATH = os.path.join(WORKING_DIR, "val_tokens.parquet")
    TEST_TOKENS_PATH = os.path.join(WORKING_DIR, "test_tokens.parquet")

    # Model Artifacts and Outputs
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # ==========================================
    # Data Hyperparameters
    # ==========================================
    VOCAB_SIZE = 50000  # Top K frequent words

    # Sequence Length:
    # Mean sentence length is ~25 words. Interleaving gaps doubles this to ~50.
    # 128 provides ample buffer for longer sentences while fitting in memory.
    MAX_LEN = 128

    # Special Tokens
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    GAP_TOKEN = "<GAP>"  # The reified gap token inserted between words

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Architecture: Transformer Encoder
    EMBED_DIM = 512
    HIDDEN_DIM = 512
    NUM_LAYERS = 8  # As per strategy
    NUM_HEADS = 8
    DROPOUT = 0.1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 64  # Optimized for A100 40GB
    LEARNING_RATE = 1e-4  # Standard for Transformers
    WEIGHT_DECAY = 1e-2
    EPOCHS = 10
    EARLY_STOPPING_PATIENCE = 3

    # Loss Weights
    # Total Loss = L_Loc + L_ID + (LAMBDA_ALIGN * L_Align)
    LAMBDA_ALIGN = 1.0  # Weight for the Latent Alignment objective

    # ==========================================
    # System & Debugging
    # ==========================================
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debugging Flags
    # Set DEBUG to True to run on a small subset of data for rapid iteration
    DEBUG = False
    DEBUG_SIZE = 10000  # Size of subset when DEBUG is True

    @classmethod
    def setup(cls):
        """
        Performs necessary setup operations, such as creating the working directory.
        Should be called at the start of execution.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)


# Automatically run setup when module is imported to ensure environment is ready
Config.setup()
