import os
import torch


class Config:
    """
    Configuration class for the RNA-Transformer Encoder project.
    Centralizes all file paths, hyperparameters, and data settings.
    """

    # =========================================================================
    # 1. File Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_3"
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata file paths
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Raw file paths (if needed)
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Checkpoint path
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Cache paths for processed tensors
    CACHE_TRAIN = os.path.join(WORKING_DIR, "train_data.pt")
    CACHE_VAL = os.path.join(WORKING_DIR, "val_data.pt")
    CACHE_TEST = os.path.join(WORKING_DIR, "test_data.pt")

    # =========================================================================
    # 2. Data Configuration
    # =========================================================================
    SEED = 42
    SEQ_LEN = 107
    PRED_LEN = 68

    # Target columns (Ground Truth)
    # Note: We predict all 5, though only 3 are scored in the metric.
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    NUM_TARGETS = len(TARGET_COLS)

    # Tokenization Maps
    # Based on vocabulary analysis: A, C, G, U
    TOKEN_MAP_SEQ = {"A": 0, "C": 1, "G": 2, "U": 3}
    VOCAB_SIZE_SEQ = len(TOKEN_MAP_SEQ)

    # Based on vocabulary analysis: (, ), .
    TOKEN_MAP_STRUCT = {".": 0, "(": 1, ")": 2}
    VOCAB_SIZE_STRUCT = len(TOKEN_MAP_STRUCT)

    # Based on vocabulary analysis: B, E, H, I, M, S, X
    TOKEN_MAP_LOOP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}
    VOCAB_SIZE_LOOP = len(TOKEN_MAP_LOOP)

    # =========================================================================
    # 3. Model Hyperparameters (Transformer Encoder)
    # =========================================================================
    # Embedding dimension for the inputs
    EMBED_DIM = 192  # Divisible by N_HEADS (e.g., 192 / 4 = 48)

    # Transformer Architecture
    N_HEADS = 4
    N_LAYERS = 4
    DIM_FEEDFORWARD = 4 * EMBED_DIM  # Standard transformer ratio
    DROPOUT = 0.1

    # Positional Encoding
    MAX_LEN = SEQ_LEN  # Fixed length

    # =========================================================================
    # 4. Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 64
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Scheduler
    WARMUP_EPOCHS = 5

    # Early Stopping
    PATIENCE = 10

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2  # For DataLoader

    @staticmethod
    def get_token_map_seq():
        return Config.TOKEN_MAP_SEQ

    @staticmethod
    def get_token_map_struct():
        return Config.TOKEN_MAP_STRUCT

    @staticmethod
    def get_token_map_loop():
        return Config.TOKEN_MAP_LOOP
