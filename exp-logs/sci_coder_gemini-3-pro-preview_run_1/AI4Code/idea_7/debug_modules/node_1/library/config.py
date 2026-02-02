import os
import torch


class Config:
    """
    Configuration for the Corrected Dual-Context Anchor Network (DC-AN).
    """

    # ==============================
    # Paths & Directories
    # ==============================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching intermediate files (embeddings) and checkpoints
    WORKING_DIR = "./working/idea_7"

    # Output directory for the final submission
    SUBMISSION_DIR = "./submission"

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Cache Paths for Precomputed Features (Parquet format)
    TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

    # Model Checkpoint
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Final Submission File
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==============================
    # Compute & Reproducibility
    # ==============================
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==============================
    # Data Hyperparameters
    # ==============================
    # Backbone Model for Text Encoding
    MODEL_BACKBONE = "sentence-transformers/all-mpnet-base-v2"

    # Tokenization Limits
    MAX_TOKEN_LEN = 128  # Max tokens per cell text for MPNet

    # Notebook Sequence Limits (for batching efficiency)
    # Truncate notebooks that exceed these counts during training
    MAX_CODE_SEQ_LEN = 200
    MAX_MD_SEQ_LEN = 100

    # ==============================
    # Model Architecture
    # ==============================
    EMBEDDING_DIM = 768  # Output dimension of all-mpnet-base-v2
    LATENT_DIM = 512  # Shared latent space dimension
    NHEAD = 8  # Number of attention heads in context transformers
    NUM_LAYERS = 2  # Number of transformer layers for context blocks
    DROPOUT = 0.1  # Dropout rate

    # ==============================
    # Training Hyperparameters
    # ==============================
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3  # No warmup, constant LR
    WEIGHT_DECAY = 0.01
    EPOCHS = 15
    PATIENCE = 3  # Early stopping patience

    # ==============================
    # Debugging / Development
    # ==============================
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SUBSET_SIZE = 2000

    @classmethod
    def setup(cls):
        """
        Ensure necessary directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories upon module import
Config.setup()
