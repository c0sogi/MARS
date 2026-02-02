import os
import torch
import random
import numpy as np


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    """
    Configuration class for the Text Normalization model.
    Centralizes hyperparameters, file paths, and system settings.
    """

    # =========================================================================
    # System & Reproducibility
    # =========================================================================
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # Adjust based on available vCPUs

    # =========================================================================
    # File Paths
    # =========================================================================
    # Input Data (Pre-generated Metadata)
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Working Directory (for caching processed data and model checkpoints)
    WORKING_DIR = "./working/idea_2"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Submission Directory
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cached Artifacts
    VOCAB_FILE = os.path.join(WORKING_DIR, "vocab.npy")
    CLASS_MAP_FILE = os.path.join(WORKING_DIR, "class_map.npy")
    MODEL_CHECKPOINT = os.path.join(WORKING_DIR, "model_checkpoint.pt")

    # Processed Data Cache (Parquet format recommended)
    TRAIN_PROCESSED = os.path.join(WORKING_DIR, "train_processed.parquet")
    VAL_PROCESSED = os.path.join(WORKING_DIR, "val_processed.parquet")
    TEST_PROCESSED = os.path.join(WORKING_DIR, "test_processed.parquet")

    # =========================================================================
    # Data Processing
    # =========================================================================
    # Column Definitions
    ID_COL = "id"
    SENTENCE_ID_COL = "sentence_id"
    TOKEN_ID_COL = "token_id"
    INPUT_COL = "before"
    TARGET_COL = "after"
    CLASS_COL = "class"

    # Special Tokens for Character-Level Tokenization
    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"
    SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

    # Token Indices
    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    UNK_IDX = 3

    # Sequence Lengths
    # Max input length observed in analysis was ~230.
    # We set a safe margin for both encoder (input) and decoder (output).
    MAX_SEQ_LEN = 300

    # Debugging / Development
    # Set to an integer (e.g., 50000) to train on a subset. Set to None for full training.
    DEBUG_SAMPLE_SIZE = None

    # =========================================================================
    # Model Architecture (Transformer Encoder-Decoder)
    # =========================================================================
    D_MODEL = 256
    NHEAD = 8
    NUM_ENCODER_LAYERS = 6
    NUM_DECODER_LAYERS = 6
    DIM_FEEDFORWARD = 1024
    DROPOUT = 0.1

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 128
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    CLIP_GRAD = 1.0

    # Auxiliary Classification Head
    # Loss = Generation_Loss + (LAMBDA_CLASS_LOSS * Classification_Loss)
    LAMBDA_CLASS_LOSS = 0.5

    # Optimization
    WARMUP_STEPS = 4000
    EARLY_STOPPING_PATIENCE = 3

    # =========================================================================
    # Inference
    # =========================================================================
    BEAM_WIDTH = 3
