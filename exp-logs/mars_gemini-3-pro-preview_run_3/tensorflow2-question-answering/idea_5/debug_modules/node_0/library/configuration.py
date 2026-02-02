import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration for the Feature-Based Boosting Ranker and Recurrent Reader pipeline.
    """

    # -------------------------------------------------------------------------
    # File Paths & Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_5"
    SUBMISSION_DIR = "./submission"

    # Source Data Files
    TRAIN_DATA_FILE = os.path.join(INPUT_DIR, "simplified-nq-train.jsonl")
    TEST_DATA_FILE = os.path.join(INPUT_DIR, "simplified-nq-test.jsonl")
    SAMPLE_SUBMISSION_FILE = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Cache File Paths (for deterministic intermediate data)
    VOCAB_CACHE_PATH = os.path.join(WORKING_DIR, "vocab.parquet")
    RANKER_TRAIN_CACHE = os.path.join(WORKING_DIR, "ranker_train_features.parquet")
    RANKER_VAL_CACHE = os.path.join(WORKING_DIR, "ranker_val_features.parquet")
    READER_TRAIN_CACHE = os.path.join(WORKING_DIR, "reader_train_data.parquet")
    READER_VAL_CACHE = os.path.join(WORKING_DIR, "reader_val_data.parquet")

    # Model Artifacts
    RANKER_MODEL_PATH = os.path.join(WORKING_DIR, "ranker_model.txt")
    READER_MODEL_PATH = os.path.join(WORKING_DIR, "reader_best.pth")

    # Final Output
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Global Settings
    # -------------------------------------------------------------------------
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4

    # -------------------------------------------------------------------------
    # Data Processing Hyperparameters
    # -------------------------------------------------------------------------
    # Preprocessing
    MIN_PARAGRAPH_LEN = 20  # Min chars to consider a text block a candidate

    # Feature Engineering (Ranker)
    BM25_K1 = 1.2
    BM25_B = 0.75
    TOP_K_RETRIEVAL = 20  # Candidates to retrieve per question
    RANKER_NEG_SAMPLES = 3  # Hard negatives per positive for training

    # Tokenization (Reader)
    VOCAB_SIZE = 30000
    UNK_TOKEN = "<UNK>"
    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"

    # Sequence Lengths
    MAX_Q_LEN = 30  # Max tokens for question
    MAX_CTX_LEN = 300  # Max tokens for candidate paragraph

    # -------------------------------------------------------------------------
    # Model Hyperparameters
    # -------------------------------------------------------------------------
    # Ranker (LightGBM)
    RANKER_PARAMS = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_jobs": 8,
        "seed": SEED,
        "force_col_wise": True,
    }
    RANKER_NUM_BOOST_ROUND = 1000
    RANKER_EARLY_STOPPING_ROUNDS = 50
    RANKER_THRESHOLD = 0.4  # Probability threshold to predict a long answer

    # Reader (Bi-GRU)
    READER_EMBEDDING_DIM = 100
    READER_HIDDEN_DIM = 128
    READER_NUM_LAYERS = 2
    READER_DROPOUT = 0.3
    READER_BIDIRECTIONAL = True

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    # Reader Training
    READER_BATCH_SIZE = 64
    READER_EPOCHS = 10
    READER_LEARNING_RATE = 0.001
    READER_PATIENCE = 3
    READER_GRAD_CLIP = 1.0

    # Dataset Subsampling (None = use all data)
    # Useful for debugging or fitting within time constraints
    TRAIN_SAMPLE_SIZE = None
    VAL_SAMPLE_SIZE = None

    @staticmethod
    def setup():
        """Ensures directories exist and sets random seeds for reproducibility."""
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

        random.seed(Config.SEED)
        np.random.seed(Config.SEED)
        torch.manual_seed(Config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(Config.SEED)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(Config.SEED)


# Initialize setup on import
Config.setup()
