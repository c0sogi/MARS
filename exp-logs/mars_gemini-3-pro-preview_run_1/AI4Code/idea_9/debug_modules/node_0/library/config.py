import os
import torch


class Config:
    # ==========================================
    # General Settings
    # ==========================================
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4

    # ==========================================
    # Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_9"
    SUBMISSION_DIR = "./submission"

    # Input Metadata
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Raw Notebook Data
    TRAIN_NB_DIR = os.path.join(INPUT_DIR, "train")
    TEST_NB_DIR = os.path.join(INPUT_DIR, "test")

    # Cached Features (Parquet)
    # We cache the MPNet embeddings to avoid re-computing them every epoch
    TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

    # Model Artifacts
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Backbone
    MODEL_CHECKPOINT = "sentence-transformers/all-mpnet-base-v2"
    EMBEDDING_DIM = 768

    # Architecture
    LATENT_DIM = 512
    NHEAD = 8
    NUM_LAYERS = 2  # Depth of the context transformers (Code Sequence & Markdown Set)
    DROPOUT = 0.1

    # ==========================================
    # Data Processing
    # ==========================================
    MAX_TOKEN_LEN = 128  # Max tokens per cell for MPNet tokenizer

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3  # No warmup, constant LR
    NUM_EPOCHS = 15
    PATIENCE = 3  # For Early Stopping

    @classmethod
    def setup(cls):
        """Ensures necessary directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Execute setup immediately when module is imported
Config.setup()
