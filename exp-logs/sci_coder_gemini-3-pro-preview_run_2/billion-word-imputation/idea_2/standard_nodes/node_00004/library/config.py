import os
import torch


class Config:
    """
    Configuration class for the Bi-Directional LSTM Word Insertion Model.
    """

    # ---------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------
    SEED = 42

    # ---------------------------------------------------------
    # File Paths & Directories
    # ---------------------------------------------------------
    # Input Metadata (Read-Only)
    METADATA_DIR = "./metadata"
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Working Directory (Read/Write for Cache & Models)
    WORKING_DIR = "./working/idea_2"
    CACHE_DIR = WORKING_DIR

    # Model Artifacts
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    TOKENIZER_PATH = os.path.join(WORKING_DIR, "tokenizer.json")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ---------------------------------------------------------
    # Data Processing Hyperparameters
    # ---------------------------------------------------------
    VOCAB_SIZE = 50000
    # Max sequence length: Mean is ~25, max is ~2000.
    # 128 covers the vast majority of sentences while keeping compute efficient.
    MAX_SEQ_LEN = 128

    # Set to an integer (e.g., 100000) to limit training data for debugging/fast prototyping.
    # Set to None to use the full dataset.
    DEBUG_SAMPLE_SIZE = None

    # ---------------------------------------------------------
    # Model Architecture
    # ---------------------------------------------------------
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 512
    LSTM_LAYERS = 2
    DROPOUT = 0.3

    # ---------------------------------------------------------
    # Training Hyperparameters
    # ---------------------------------------------------------
    # Batch size optimized for A100 40GB
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 3
    WEIGHT_DECAY = 1e-5
    CLIP_GRAD = 5.0

    # Early Stopping
    PATIENCE = 2

    # Loss Weighting
    # Total Loss = Location_Loss + (LOSS_LAMBDA * Word_Generation_Loss)
    LOSS_LAMBDA = 1.0

    # ---------------------------------------------------------
    # System / Hardware
    # ---------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 12  # Utilizing available vCPUs

    @classmethod
    def setup(cls):
        """Creates necessary directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print("=" * 40)
        print(f"{'CONFIG':^40}")
        print("=" * 40)
        for key, value in cls.__dict__.items():
            if not key.startswith("__") and not callable(value):
                print(f"{key:<25} : {value}")
        print("=" * 40)


# Initialize directories on import
Config.setup()
