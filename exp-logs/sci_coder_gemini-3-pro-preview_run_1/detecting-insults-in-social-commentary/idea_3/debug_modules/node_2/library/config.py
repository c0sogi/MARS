import os
import torch


class Config:
    """
    Configuration class for the Hybrid DeBERTa-v3 with Structural Feature Fusion.
    """

    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # File Paths
    # ==========================================
    # Input Metadata (Generated in previous steps)
    METADATA_DIR = "./metadata"
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Directories
    WORKING_DIR = "./working/idea_3"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Model Architecture
    # ==========================================
    MODEL_NAME = "microsoft/deberta-v3-base"
    MAX_LENGTH = 256  # Covers the majority of comment lengths based on EDA
    HIDDEN_SIZE = 768  # Standard hidden size for base transformers

    # ==========================================
    # Structural Branch (Feature Engineering)
    # ==========================================
    # Character N-grams for robustness against obfuscation
    NGRAM_RANGE = (3, 5)
    # Dimensionality reduction for the structural features
    SVD_COMPONENTS = 128
    # Max features for TF-IDF before SVD
    TFIDF_MAX_FEATURES = 50000

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 16
    NUM_EPOCHS = 5
    PATIENCE = 2  # For Early Stopping

    # Differential Learning Rates
    # Lower rate for pre-trained backbone to preserve knowledge
    LR_BACKBONE = 2e-5
    # Higher rate for the new fusion head to learn quickly
    LR_HEAD = 1e-3

    WEIGHT_DECAY = 0.01
    DROPOUT_RATE = 0.2  # High dropout for fusion head regularization

    # ==========================================
    # Hardware
    # ==========================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 2
