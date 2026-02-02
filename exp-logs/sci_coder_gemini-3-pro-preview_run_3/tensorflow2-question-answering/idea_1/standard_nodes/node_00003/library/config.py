import os
import random
import numpy as np
import torch


class Config:
    """
    Configuration parameters for the Hierarchical Embed-and-Scan Network pipeline.
    """

    # --- Paths ---
    INPUT_DIR = "./input"
    TRAIN_FILE = os.path.join(INPUT_DIR, "simplified-nq-train.jsonl")
    TEST_FILE = os.path.join(INPUT_DIR, "simplified-nq-test.jsonl")
    SAMPLE_SUBMISSION_FILE = os.path.join(INPUT_DIR, "sample_submission.csv")

    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Cache directory for deterministic data processing
    CACHE_DIR = "./working/idea_1/"

    # Output directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Reproducibility ---
    SEED = 42

    # --- Data Processing Hyperparameters ---
    # Debugging: Set to a small integer (e.g., 5000) to limit dataset size for fast testing.
    # Set to None to use the full dataset.
    MAX_TRAIN_SAMPLES = None
    MAX_VAL_SAMPLES = None

    # Vocabulary
    MAX_VOCAB_SIZE = 30000
    MIN_FREQ = 2
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    # Sequence Lengths
    MAX_Q_LEN = 30  # Maximum tokens for questions
    MAX_CTX_LEN = (
        400  # Maximum tokens for candidate paragraphs (Long Answer candidates)
    )

    # Candidate Generation
    # HTML tags used to split documents into candidate blocks
    SPLIT_TAGS = ["<P>", "<Table>", "<Ul>", "<Ol>", "<Dl>"]

    # --- Model Hyperparameters ---
    EMBEDDING_DIM = 100  # Dimension for word embeddings

    # Long Answer Ranker (Siamese DAN)
    RANKER_HIDDEN_DIM = 256
    RANKER_NUM_LAYERS = 2
    RANKER_DROPOUT = 0.3

    # Short Answer Reader (Shallow CNN)
    READER_FILTERS = 128
    READER_KERNEL_SIZES = [3, 5]
    READER_DROPOUT = 0.3

    # --- Training Hyperparameters ---
    BATCH_SIZE = 64
    NUM_WORKERS = 4  # Number of dataloader workers
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    EPOCHS = 10

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 3
    EARLY_STOPPING_DELTA = 0.001

    # Ranking Loss
    NUM_NEGATIVES = 4  # Number of negative samples per positive sample for ranking
    RANKING_MARGIN = 0.5  # Margin for MarginRankingLoss

    # --- Inference Hyperparameters ---
    # Threshold for the ranker score. If top score < threshold, predict NULL.
    CONFIDENCE_THRESHOLD = 0.1

    @staticmethod
    def setup():
        """
        Initializes the environment: creates directories and sets random seeds.
        """
        # Ensure directories exist
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

        # Set fixed random seeds for reproducibility
        random.seed(Config.SEED)
        np.random.seed(Config.SEED)
        torch.manual_seed(Config.SEED)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(Config.SEED)
            # Ensure deterministic behavior in CuDNN
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @staticmethod
    def get_device():
        """Returns the appropriate torch device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Execute setup immediately upon import
Config.setup()
