import os
import torch


class Config:
    # ==========================================
    # Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Specific working directory for this idea (idea_26)
    WORKING_DIR = "./working/idea_26"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Cache file paths
    CACHE_TRAIN_PATH = os.path.join(WORKING_DIR, "train_processed.parquet")
    CACHE_VAL_PATH = os.path.join(WORKING_DIR, "val_processed.parquet")
    CACHE_TEST_PATH = os.path.join(WORKING_DIR, "test_processed.parquet")
    CACHE_VOCAB_PATH = os.path.join(WORKING_DIR, "vocab_sizes.npy")

    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Adjust based on CPU availability

    # ==========================================
    # Data Hyperparameters
    # ==========================================
    # Embedding dimension for all categorical features
    EMBEDDING_DIM = 16

    # Feature Engineering
    # f_27 decomposition results in 10 character columns + 1 unique count column
    # Plus original normalized continuous features

    # ==========================================
    # Model Architecture (RPFE)
    # ==========================================
    NUM_STREAMS = 5

    # Stream Configurations:
    # Streams 1 & 2 (Anchors): Standard Funnel, Dropout 0.20
    # Stream 3 (Capacity Variant): Wide Funnel, Dropout 0.25
    # Stream 4 (Conservative Variant): Standard Funnel, Dropout 0.25
    # Stream 5 (Conservative Variant): Standard Funnel, Dropout 0.30

    # Layer dimensions
    LAYERS_STANDARD = [512, 256, 128]
    LAYERS_WIDE = [1024, 512, 256]

    # Configuration for each stream [hidden_layers, dropout_rate]
    STREAM_CONFIGS = [
        {"layers": LAYERS_STANDARD, "dropout": 0.20},  # Stream 1
        {"layers": LAYERS_STANDARD, "dropout": 0.20},  # Stream 2
        {"layers": LAYERS_WIDE, "dropout": 0.25},  # Stream 3
        {"layers": LAYERS_STANDARD, "dropout": 0.25},  # Stream 4
        {"layers": LAYERS_STANDARD, "dropout": 0.30},  # Stream 5
    ]

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 1024
    EPOCHS = 50

    # Optimizer (AdamW)
    WEIGHT_DECAY = 1e-4

    # Scheduler (OneCycleLR)
    MAX_LR = 1e-2
    PCT_START = 0.3  # Default for OneCycle, can be tuned
    DIV_FACTOR = 25.0
    FINAL_DIV_FACTOR = 10000.0

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 10

    def __init__(self):
        pass
