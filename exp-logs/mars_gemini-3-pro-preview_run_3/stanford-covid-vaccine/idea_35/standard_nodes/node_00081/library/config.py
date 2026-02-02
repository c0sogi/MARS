import os
import torch


class Config:
    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for Idea 35 (Deep Decoupled Channel-Gated BiGRU)
    WORKING_DIR = "./working/idea_35"
    SUBMISSION_DIR = "./submission"

    # Input Files (Parquet Metadata)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Output Files
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Specifications
    # =========================================================================
    SEQ_LEN = 107
    PRED_LEN = 68  # seq_scored

    # Input Features:
    # 4 (Sequence: A,G,U,C) + 3 (Structure: (,),.) + 7 (Loop: S,M,I,B,H,E,X)
    INPUT_DIM = 14

    # Target Columns in the dataset
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    NUM_TARGETS = len(TARGET_COLS)

    # Columns used for the official metric (MCRMSE)
    # Note: deg_pH10 and deg_50C are predicted but not scored
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # Indices of scored columns within TARGET_COLS for slicing tensors
    # reactivity(0), deg_Mg_pH10(1), deg_pH10(2), deg_Mg_50C(3), deg_50C(4)
    SCORED_COLS_INDICES = [0, 1, 3]

    # =========================================================================
    # Model Hyperparameters (DDCG-BiGRU)
    # =========================================================================
    # Convolutional Stem
    CONV_FILTERS = 256
    CONV_KERNEL = 3

    # Deep Backbone
    HIDDEN_SIZE = 384
    NUM_LAYERS = 4
    DROPOUT = 0.1

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Optimization
    EPOCHS = 50
    PATIENCE = 7  # For Early Stopping

    # Stability (Critical for Deep RNNs)
    GRADIENT_CLIP = 1.0

    # Scheduler
    MIN_LR = 1e-6
    T_MAX = EPOCHS  # For Cosine Annealing

    # =========================================================================
    # System & Reproducibility
    # =========================================================================
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debugging
    DEBUG = False
    DEBUG_SAMPLES = 100

    @classmethod
    def setup(cls):
        """Ensures necessary directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
