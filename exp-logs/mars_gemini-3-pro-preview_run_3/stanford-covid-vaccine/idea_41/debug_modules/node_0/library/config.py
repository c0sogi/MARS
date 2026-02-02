import os
import torch


class Config:
    # ==========================================
    # Directories & Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_41"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # File Paths
    TRAIN_PARQUET = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PARQUET = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PARQUET = os.path.join(METADATA_DIR, "test.parquet")

    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")
    SUBMISSION_FILE = os.path.join(WORKING_DIR, "submission.csv")
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # ==========================================
    # Data Parameters
    # ==========================================
    SEQ_LEN = 107
    PRED_LEN = 68

    # Input Channels:
    # 4 (One-Hot Nucleotide: A, G, C, U)
    # + 3 (One-Hot Structure: (, ), .)
    # + 7 (One-Hot Loop Type: S, M, I, B, H, E, X)
    NUM_INPUT_CHANNELS = 14

    # Targets
    NUM_TARGETS = 5
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # Metric is calculated only on these columns
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Architecture: Deep Decoupled BiGRU with Bias-Driven Loop Refinement
    HIDDEN_DIM = 384
    NUM_LAYERS = 4
    DROPOUT = 0.1

    # Convolutional Stem
    STEM_KERNEL_SIZE = 3
    STEM_FILTERS = 256

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    MAX_EPOCHS = 50
    CLIP_GRAD_NORM = 1.0
    PATIENCE = 10  # For Early Stopping

    # Scheduler (Cosine Annealing)
    T_MAX = MAX_EPOCHS
    ETA_MIN = 1e-6

    # ==========================================
    # System & Debugging
    # ==========================================
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debug Mode: Train on a small subset to verify pipeline
    DEBUG = False
    DEBUG_SUBSET_SIZE = 100
