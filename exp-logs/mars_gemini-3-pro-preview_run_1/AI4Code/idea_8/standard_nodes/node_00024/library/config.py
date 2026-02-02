import os
import torch


class Config:
    # ==========================================
    # General Settings
    # ==========================================
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for Idea 8 (Corrected Dual-Context Anchor Network)
    WORKING_DIR = "./working/idea_8"
    os.makedirs(WORKING_DIR, exist_ok=True)

    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Feature Cache Paths (Parquet)
    TRAIN_FEATS_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATS_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_FEATS_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

    # Model Checkpoint
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Final Submission
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Backbone
    MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
    EMBEDDING_DIM = 768  # Output dim of MPNet

    # DC-AN Architecture
    LATENT_DIM = 512  # Projection dimension
    NHEAD = 8  # Transformer heads
    NUM_LAYERS = 2  # Transformer layers for context encoding
    DROPOUT = 0.1

    # Sequence Handling
    MAX_SEQ_LEN = 256  # Maximum number of cells (code + md) to process per notebook

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 10  # Upper bound, relies on early stopping
    PATIENCE = 3  # Early stopping patience
    WARMUP_RATIO = 0.0  # Explicitly disabled as per instructions
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0

    def __init__(self, **kwargs):
        """
        Allow overriding configuration defaults via keyword arguments.
        """
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
