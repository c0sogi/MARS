import os
import torch


class Config:
    # ==========================================
    # 1. Reproducibility & Hardware
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Adjust based on available vCPUs (12 available)

    # ==========================================
    # 2. Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_5"
    SUBMISSION_DIR = "./submission"

    # Metadata Sources
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Feature Cache Paths (Parquet format)
    # These store the pre-computed MPNet embeddings
    TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

    # Model & Output Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 3. Model Architecture Hyperparameters
    # ==========================================
    # Backbone: Similarity-tuned MPNet
    BACKBONE_NAME = "sentence-transformers/all-mpnet-base-v2"
    INPUT_DIM = 768  # Dimension of MPNet embeddings

    # Dual-Context Network
    HIDDEN_DIM = 512  # Shared latent space dimension
    NHEAD = 8  # Attention heads for context transformers
    NUM_ENCODER_LAYERS = 2  # Layers for Code (Seq) and Markdown (Set) transformers
    DIM_FEEDFORWARD = 2048
    DROPOUT = 0.1

    # ==========================================
    # 4. Data Processing
    # ==========================================
    MAX_LENGTH = 128  # Max token length for backbone encoding

    # Debugging / Development
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 1000  # Number of notebooks to use when DEBUG is True

    # ==========================================
    # 5. Training Strategy
    # ==========================================
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3  # Constant LR, no warmup
    WEIGHT_DECAY = 0.01
    EPOCHS = 10
    PATIENCE = 3  # Early stopping patience

    @classmethod
    def setup(cls):
        """
        Creates necessary directories for caching and submission.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Ensure reproducibility in hash-based operations
        os.environ["PYTHONHASHSEED"] = str(cls.SEED)
