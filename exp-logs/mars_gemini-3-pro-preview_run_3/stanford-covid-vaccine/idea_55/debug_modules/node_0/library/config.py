import os
import torch


class Config:
    # ==========================================
    # System & Paths
    # ==========================================
    PROJECT_NAME = "RNA_Degradation_Prediction"
    SEED = 42

    # Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_55"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # File Paths
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # Cache Paths (for data processing)
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_cache.npz")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_cache.npz")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_cache.npz")

    # ==========================================
    # Data Dimensions & Features
    # ==========================================
    SEQ_LEN = 107
    SEQ_SCORED = 68

    # Input Features (One-Hot Encoding)
    # 4 Nucleotides (A, G, C, U)
    # 3 Structure characters ( (, ), . )
    # 7 Predicted Loop Types (S, M, I, B, H, E, X)
    INPUT_DIM = 4 + 3 + 7  # 14

    # Targets
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    NUM_TARGETS = len(TARGET_COLS)

    # Scored Targets (for Validation Metric)
    SCORED_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    SCORED_INDICES = [0, 1, 3]  # Indices corresponding to TARGET_COLS

    # ==========================================
    # Model Hyperparameters (SDBR-BiGRU)
    # ==========================================
    # 1D Convolutional Stem
    CONV_FILTERS = 256
    CONV_KERNEL_SIZE = 3

    # Backbone (BiGRU)
    HIDDEN_DIM = 384
    NUM_LAYERS = 3  # Validated optimal depth
    DROPOUT = 0.1

    # Structural Interaction Module
    # Uses decoupled message passing with bias refinement

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Gradient Clipping (Mandatory for hybrid architecture)
    MAX_GRAD_NORM = 1.0

    # Scheduler (Cosine Annealing)
    T_MAX = NUM_EPOCHS
    ETA_MIN = 1e-6

    # Early Stopping
    PATIENCE = 10

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    @staticmethod
    def print_config():
        print(f"==== Configuration ====")
        print(f"Device: {Config.DEVICE}")
        print(f"Working Dir: {Config.WORKING_DIR}")
        print(f"Model: {Config.NUM_LAYERS}-Layer BiGRU (Hidden: {Config.HIDDEN_DIM})")
        print(f"Input Dim: {Config.INPUT_DIM}, Output Dim: {Config.NUM_TARGETS}")
        print(
            f"Training: BS={Config.BATCH_SIZE}, LR={Config.LEARNING_RATE}, Epochs={Config.NUM_EPOCHS}"
        )
        print(f"=======================")
