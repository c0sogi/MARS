import os


class Config:
    """
    Configuration for the Interaction Map Convolutional Network (IMCN) pipeline.
    """

    # ==========================================
    # Paths and Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_6"
    # Changed to subdirectory to match the path expected by NoSubmissionError
    SUBMISSION_DIR = "submission"

    # Source Data Files
    TRAIN_DATA_FILE = "simplified-nq-train.jsonl"
    TEST_DATA_FILE = "simplified-nq-test.jsonl"
    SAMPLE_SUBMISSION_FILE = "sample_submission.csv"

    # Metadata Files (Parquet)
    TRAIN_META_FILE = "train.parquet"
    VAL_META_FILE = "val.parquet"
    TEST_META_FILE = "test.parquet"

    # Output Files
    SUBMISSION_FILE = "submission.csv"
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "imcn_model.pth")
    VOCAB_SAVE_PATH = os.path.join(WORKING_DIR, "vocab.json")

    # ==========================================
    # Data Preprocessing Hyperparameters
    # ==========================================
    # Sequence Lengths
    MAX_Q_LEN = 32  # Fixed length for Question tokens
    MAX_C_LEN = 300  # Fixed length for Candidate Answer tokens

    # Vocabulary and Embeddings
    VOCAB_SIZE = 15000  # Maximum number of tokens in vocabulary
    EMBED_DIM = 64  # Dimension of token embeddings
    UNK_TOKEN = "<UNK>"
    PAD_TOKEN = "<PAD>"

    # Debugging
    # Set to a integer (e.g., 1000) to limit dataset size for quick debugging.
    # Set to None for full training.
    DEBUG_SAMPLE_SIZE = None

    # ==========================================
    # Model Architecture Hyperparameters
    # ==========================================
    NUM_FILTERS = 32  # Number of filters in the 2D Convolutional layer
    KERNEL_SIZE = 3  # Size of the 2D Convolution kernel (e.g., 3x3)
    HIDDEN_DIM = 128  # Dimension of the dense layer after pooling
    DROPOUT_RATE = 0.3  # Dropout probability

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 5

    # Negative Sampling
    # Ratio of negative examples (candidates that are not the answer) to keep during training.
    # 1.0 means keep all, 0.2 means keep 20% of negatives.
    NEGATIVE_SAMPLING_RATIO = 0.2

    # Loss Weights
    # Total Loss = W_LONG * Long_Loss + W_SHORT * Short_Loss
    WEIGHT_LONG = 1.0
    WEIGHT_SHORT = 1.0

    # Early Stopping
    PATIENCE = 2  # Number of epochs to wait for validation improvement

    # Reproducibility
    SEED = 42

    # ==========================================
    # Inference Thresholds
    # ==========================================
    # Threshold for predicting a Long Answer exists (Sigmoid output)
    TAU_LONG = 0.4

    # Threshold for predicting a Short Answer exists within the Long Answer
    # This can be applied to the confidence of the best span
    TAU_SHORT = 0.4

    @classmethod
    def ensure_directories(cls):
        """
        Ensures that necessary working and submission directories exist.
        """
        if not os.path.exists(cls.WORKING_DIR):
            os.makedirs(cls.WORKING_DIR, exist_ok=True)
        if not os.path.exists(cls.SUBMISSION_DIR):
            os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
