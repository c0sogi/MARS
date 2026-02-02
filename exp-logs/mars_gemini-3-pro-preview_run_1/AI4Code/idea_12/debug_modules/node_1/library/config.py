import os


class Config:
    """
    Configuration class for the Corrected Dual-Context Anchor Network (DC-AN).
    Centralizes file paths, model hyperparameters, and training settings.
    """

    # ==========================================
    # 1. DIRECTORY SETUP
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_12"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # ==========================================
    # 2. DATA PATHS
    # ==========================================
    # Metadata files (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Cache files (Parquet format for embeddings)
    TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

    # Output paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # ==========================================
    # 3. MODEL HYPERPARAMETERS
    # ==========================================
    # Backbone
    MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

    # Dimensions
    INPUT_DIM = 768  # Output dimension of MPNet
    LATENT_DIM = 512  # Shared latent space dimension

    # Context Transformer Settings (Used for both Code Sequence and Markdown Set)
    N_HEADS = 8
    NUM_LAYERS = 2
    DROPOUT = 0.1

    # ==========================================
    # 4. TRAINING SETTINGS
    # ==========================================
    SEED = 42
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3  # No warmup, constant LR
    NUM_EPOCHS = 5
    NUM_WORKERS = 4

    # ==========================================
    # 5. DEBUGGING / DEVELOPMENT
    # ==========================================
    # If True, only processes a small subset of data for rapid testing
    DEBUG = False
    DEBUG_SAMPLES = 2000
