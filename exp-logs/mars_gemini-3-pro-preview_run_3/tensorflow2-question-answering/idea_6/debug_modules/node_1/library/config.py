import os
import torch


class PathConfig:
    """
    Configuration for file paths and directories.
    """

    # Base directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_6"
    SUBMISSION_DIR = "./submission"

    # Input data files
    TRAIN_FILE = os.path.join(INPUT_DIR, "simplified-nq-train.jsonl")
    TEST_FILE = os.path.join(INPUT_DIR, "simplified-nq-test.jsonl")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata files
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output/Cache files
    RANKER_TRAIN_DATA = os.path.join(WORKING_DIR, "ranker_train_data.parquet")
    RANKER_VAL_DATA = os.path.join(WORKING_DIR, "ranker_val_data.parquet")
    READER_TRAIN_DATA = os.path.join(WORKING_DIR, "reader_train_data.parquet")
    READER_VAL_DATA = os.path.join(WORKING_DIR, "reader_val_data.parquet")

    # Model checkpoints
    RANKER_MODEL_PATH = os.path.join(WORKING_DIR, "ranker_best.pth")
    READER_MODEL_PATH = os.path.join(WORKING_DIR, "reader_best.pth")

    # Submission output
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    @staticmethod
    def ensure_dirs():
        """Creates necessary output directories."""
        os.makedirs(PathConfig.WORKING_DIR, exist_ok=True)
        os.makedirs(PathConfig.SUBMISSION_DIR, exist_ok=True)


class ModelConfig:
    """
    Configuration for model architecture and tokenization.
    """

    # Backbone model (Distilled Transformer)
    MODEL_NAME = "microsoft/MiniLM-L12-H384-uncased"

    # Tokenization settings
    MAX_Q_LEN = 64  # Maximum length for question tokens
    MAX_CTX_LEN = 384  # Maximum length for context/paragraph tokens
    DOC_STRIDE = 128  # Overlap for splitting long documents (if needed)

    # Ranker specific
    RANKER_POOLING = "mean"  # Pooling strategy for ranker (mean or cls)

    # Reader specific
    READER_HIDDEN_SIZE = 384  # Should match backbone hidden size

    # Inference thresholds
    RANKER_THRESHOLD = 0.5  # Threshold for considering a paragraph relevant
    SHORT_ANSWER_THRESHOLD = 0.1  # Threshold for short answer confidence


class TrainingConfig:
    """
    Configuration for training hyperparameters and optimization.
    """

    # General
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # Data sampling (for debugging/speed)
    # Set to an integer (e.g., 1000) to limit dataset size, or None for full dataset
    SUBSET_SIZE = None

    # Optimization
    BATCH_SIZE = 32
    LEARNING_RATE = 3e-5
    WEIGHT_DECAY = 0.01
    EPOCHS = 3

    # Early Stopping
    EARLY_STOPPING_PATIENCE = (
        2  # Number of epochs to wait before stopping if no improvement
    )

    # Loss weights (if applicable)
    POSITIVE_WEIGHT = 1.0
    NEGATIVE_WEIGHT = 1.0
