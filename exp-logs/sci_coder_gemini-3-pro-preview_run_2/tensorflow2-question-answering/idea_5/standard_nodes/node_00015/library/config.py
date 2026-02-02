import os
import torch


class Config:
    """
    Global configuration for the Feed-Forward Decomposable Attention Network pipeline.
    """

    # --------------------------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------------------------
    SEED = 42

    # --------------------------------------------------------------------------
    # File Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_5"
    SUBMISSION_DIR = "./submission"

    # Raw Data
    TRAIN_DATA_PATH = os.path.join(INPUT_DIR, "simplified-nq-train.jsonl")
    TEST_DATA_PATH = os.path.join(INPUT_DIR, "simplified-nq-test.jsonl")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata (Generated previously)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "validation_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Cache Files (Parquet/NPY as required)
    VOCAB_PATH = os.path.join(WORKING_DIR, "vocab.npy")
    EMBEDDING_MATRIX_PATH = os.path.join(WORKING_DIR, "embedding_matrix.npy")

    # Model Checkpoints
    MODEL_CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Final Submission
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Processing Hyperparameters
    # --------------------------------------------------------------------------
    # Sequence Lengths
    Q_MAX_LEN = 20  # Maximum tokens for Question
    C_MAX_LEN = 300  # Maximum tokens for Candidate text

    # Vocabulary
    VOCAB_SIZE = 50000  # Max vocabulary size
    EMBED_DIM = 100  # Dimension of word embeddings
    UNK_TOKEN = "<UNK>"
    PAD_TOKEN = "<PAD>"

    # Debugging
    # Set to a small integer (e.g., 1000) to limit dataset size for debugging.
    # Set to None to use the full dataset.
    DEBUG_SAMPLE_SIZE = None

    # --------------------------------------------------------------------------
    # Model Architecture Hyperparameters
    # --------------------------------------------------------------------------
    HIDDEN_DIM = 256
    DROPOUT = 0.3
    NUM_CLASSES_YN = 3  # YES, NO, NONE

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 10
    EARLY_STOPPING_PATIENCE = 2

    # Negative Sampling: Number of negative candidates to sample per positive one
    # If a document has no positive, we might skip it or sample negatives only.
    # For this pipeline, we assume balanced sampling in the dataloader.
    NEG_SAMPLE_RATIO = 1

    # --------------------------------------------------------------------------
    # Inference Hyperparameters
    # --------------------------------------------------------------------------
    # Score threshold for predicting a long answer.
    # If the max score for a document is below this, prediction is BLANK.
    CONFIDENCE_THRESHOLD = 0.5

    # --------------------------------------------------------------------------
    # Hardware
    # --------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self):
        """
        Initialize configuration and ensure necessary directories exist.
        """
        os.makedirs(self.WORKING_DIR, exist_ok=True)
        os.makedirs(self.SUBMISSION_DIR, exist_ok=True)

    def display(self):
        """Prints the configuration."""
        print("=== Configuration ===")
        print(f"Device: {self.DEVICE}")
        print(f"Working Directory: {self.WORKING_DIR}")
        print(f"Batch Size: {self.BATCH_SIZE}")
        print(f"Learning Rate: {self.LEARNING_RATE}")
        print(f"Max Question Len: {self.Q_MAX_LEN}")
        print(f"Max Candidate Len: {self.C_MAX_LEN}")
        print(f"Debug Sample Size: {self.DEBUG_SAMPLE_SIZE}")
        print("=====================")
