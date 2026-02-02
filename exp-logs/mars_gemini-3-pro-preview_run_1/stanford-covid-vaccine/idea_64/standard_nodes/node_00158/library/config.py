import os
import torch


class Config:
    # General
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2  # Adjust based on available vCPUs

    # Data Paths
    METADATA_DIR = "./metadata"
    TRAIN_FILE = os.path.join(METADATA_DIR, "train.parquet")
    VAL_FILE = os.path.join(METADATA_DIR, "val.parquet")
    TEST_FILE = os.path.join(METADATA_DIR, "test.parquet")
    SAMPLE_SUBMISSION = "./input/sample_submission.csv"

    # Caching
    CACHE_DIR = "./working/idea_64/"
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Data Dimensions
    SEQ_LEN = 107
    PRED_LEN = 68

    # Vocab Sizes
    SEQ_VOCAB_SIZE = 4  # A, G, C, U
    LOOP_VOCAB_SIZE = 7  # B, E, H, I, M, S, X

    # Model Hyperparameters
    # Embeddings
    SEQ_EMBED_DIM = 128
    LOOP_EMBED_DIM = 64
    DIST_EMBED_DIM = 64

    # Backbone
    HIDDEN_SIZE = 512  # Wide stream width
    NUM_LAYERS = 6  # Number of residual blocks
    DROPOUT = 0.1  # Dropout applied in residual blocks

    # Output
    NUM_CLASSES = 3  # reactivity, deg_Mg_pH10, deg_Mg_50C

    # Training Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    MAX_GRAD_NORM = 1.0
    EPOCHS = 20

    # Scheduler
    T_MAX = EPOCHS  # For Cosine Annealing

    # Columns
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    # Columns present in data but not used for training/scoring in this specific idea
    IGNORED_TARGETS = ["deg_pH10", "deg_50C"]
