import os
import torch


class Config:
    """
    Configuration for the Dual-Context Anchor Network (DC-AN) project.
    """

    # ==========================================
    # 1. Reproducibility & Debugging
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data
    DEBUG_SAMPLE_SIZE = 1000  # Number of samples to use in debug mode

    # ==========================================
    # 2. File Paths & Directories
    # ==========================================
    # Root Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_6"
    SUBMISSION_DIR = "./submission"

    # Metadata Files (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Cache Files (Parquet format for embeddings)
    TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

    # Model & Submission Outputs
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 3. Model Hyperparameters
    # ==========================================
    # Backbone
    BACKBONE_NAME = "sentence-transformers/all-mpnet-base-v2"
    FREEZE_BACKBONE = True  # Backbone is frozen, we use pre-computed embeddings

    # Dimensions
    INPUT_DIM = 768  # Output dimension of MPNet
    HIDDEN_DIM = 512  # Projection dimension for the shared latent space

    # Sequence Processing
    MAX_LENGTH = 128  # Max token length for generating embeddings

    # Context Transformers
    NHEAD = 8
    NUM_LAYERS = 2

    # ==========================================
    # 4. Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 15
    EARLY_STOPPING_PATIENCE = 3

    # Optimization
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 0  # Explicitly disabled per solution design
    LABEL_SMOOTHING = 0.0  # Explicitly excluded per solution design
    GRADIENT_CLIP = 1.0

    # ==========================================
    # 5. System & Compute
    # ==========================================
    NUM_WORKERS = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def setup(cls):
        """
        Ensures necessary working directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Automatically create directories when config is imported
Config.setup()
