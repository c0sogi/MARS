import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration for the Multi-Task Neuro-Symbolic Cascade (Idea 6).
    """

    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_6"

    # Input Data (Parquet files from metadata)
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Output/Cache Directories
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    STATS_DIR = os.path.join(WORKING_DIR, "stats")
    SUBMISSION_DIR = "./submission"

    # File Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "model_best.pt")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Processing Hyperparameters
    # ==========================================
    # Context window: [prev] <SEP> [target] <SEP> [next]
    CONTEXT_WINDOW_SIZE = 1

    # Sequence Lengths
    # Max length for the character sequence of the input (including context)
    MAX_INPUT_LEN = 128
    # Max length for the generated output text
    MAX_OUTPUT_LEN = 128

    # Debugging / Development
    DEBUG = False
    DEBUG_SIZE = 50000  # Number of samples to use if DEBUG is True

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Architecture: Bi-GRU Encoder + Uni-GRU Decoder + Attention + Aux Head
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 512
    ENC_LAYERS = 2
    DEC_LAYERS = 1
    DROPOUT = 0.3

    # Auxiliary Classification Head
    # Data analysis shows 16 unique classes. We add 1 for padding/unknown.
    NUM_AUX_CLASSES = 17

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 512
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 20
    PATIENCE = 3  # Early stopping patience
    TEACHER_FORCING_RATIO = 0.5
    CLIP_GRAD = 1.0

    # Multi-Task Loss Weighting
    # L_total = L_gen + LAMBDA_AUX * L_class
    LAMBDA_AUX = 0.5

    # ==========================================
    # Special Tokens & Vocabulary
    # ==========================================
    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"
    SEP_TOKEN = "<sep>"

    # Reserved Indices
    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    UNK_IDX = 3
    SEP_IDX = 4

    # ==========================================
    # Compute
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    @classmethod
    def setup_environment(cls):
        """
        Sets up the directory structure and random seeds for reproducibility.
        """
        # Create directories
        for path in [cls.WORKING_DIR, cls.CACHE_DIR, cls.STATS_DIR, cls.SUBMISSION_DIR]:
            os.makedirs(path, exist_ok=True)

        # Set seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        print(f"Environment setup complete. Device: {cls.DEVICE}")
        print(f"Working directory: {cls.WORKING_DIR}")
