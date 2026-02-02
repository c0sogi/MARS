import os
import torch


class Config:
    """
    Global configuration for Idea 5: Transformer-CRF Sequence Labeling.
    Handles paths, hyperparameters, and data processing settings.
    """

    # ==========================================
    # 1. General & Paths
    # ==========================================
    IDEA_NAME = "idea_5"
    SEED = 42

    # Input Directories (Read-Only)
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working Directory (Write Allowed)
    WORKING_DIR = os.path.join("./working", IDEA_NAME)

    # Metadata File Paths
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Cache File Paths (Parquet for data, Bin for model)
    TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_processed.parquet")
    VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_processed.parquet")
    TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_processed.parquet")
    MODEL_CHECKPOINT_PATH = os.path.join(WORKING_DIR, "model_checkpoint.bin")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # ==========================================
    # 2. Model Hyperparameters
    # ==========================================
    # Backbone model for contextual embeddings
    MODEL_NAME = "roberta-base"

    # Sequence Length: Analysis showed max sentence length is ~256 tokens.
    # Subword tokenization increases this, but 256 covers most cases efficiently.
    MAX_LEN = 256

    # Training
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 64
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0

    # ==========================================
    # 3. Data Processing & Sampling
    # ==========================================
    # Strategic Sampling:
    # The dataset is ~93% PLAIN/PUNCT. We keep all "interesting" sentences
    # (containing at least one non-PLAIN token) and subsample the trivial ones.
    TRIVIAL_PLAIN_KEEP_RATE = 0.05  # Keep 5% of purely PLAIN sentences

    # Debugging
    DEBUG = False  # Set to True to run on a tiny subset
    DEBUG_SAMPLE_SIZE = 5000

    # ==========================================
    # 4. Labels / Classes
    # ==========================================
    # Standard Text Normalization Classes (16 classes)
    LABELS = [
        "PLAIN",
        "PUNCT",
        "DATE",
        "LETTERS",
        "CARDINAL",
        "VERBATIM",
        "MEASURE",
        "ORDINAL",
        "DECIMAL",
        "MONEY",
        "DIGIT",
        "ELECTRONIC",
        "TELEPHONE",
        "TIME",
        "FRACTION",
        "ADDRESS",
    ]

    NUM_LABELS = len(LABELS)
    LABEL2ID = {label: i for i, label in enumerate(LABELS)}
    ID2LABEL = {i: label for i, label in enumerate(LABELS)}

    # ==========================================
    # 5. Hardware
    # ==========================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 12 vCPUs available; 4 workers is usually a safe sweet spot for PyTorch DataLoaders
    NUM_WORKERS = 4

    @classmethod
    def setup(cls):
        """Ensures the working directory exists."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
