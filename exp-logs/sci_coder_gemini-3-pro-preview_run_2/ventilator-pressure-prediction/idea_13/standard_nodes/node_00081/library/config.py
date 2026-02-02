import os
import torch


class Config:
    """
    Configuration class for the Ventilator Pressure Prediction task.
    Defines file paths, model hyperparameters, training settings, and data processing flags.
    """

    # ==========================================
    # System & Reproducibility
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Utilizing available vCPUs for data loading
    NUM_WORKERS = 12

    # ==========================================
    # Directories & Files
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_13"
    SUBMISSION_DIR = "./submission"

    # Ensure output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Raw Data Paths
    TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
    TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Paths (Generated previously)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Cache Paths (for deterministic data processing)
    # Using .parquet for dataframes and .npy for scaler statistics
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_processed.parquet")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_processed.parquet")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_processed.parquet")
    SCALER_CACHE = os.path.join(WORKING_DIR, "scaler_params.npy")

    # Model Checkpoint Path
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Submission Output
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Model Hyperparameters (DGC-BiLSTM)
    # ==========================================
    # Deep Gated-Cascade BiLSTM Architecture
    HIDDEN_DIM = 512
    INJECTION_DIM = 128
    NUM_LAYERS = 4
    DROPOUT = 0.1
    BIDIRECTIONAL = True

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    # Stretched-Horizon Convergence Protocol
    EPOCHS = 200
    BATCH_SIZE = 256  # Optimized for A100 GPU
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    CLIP_GRAD = 1.0

    # Scheduler (Cosine Annealing)
    SCHEDULER_T_MAX = 180
    SCHEDULER_ETA_MIN = 1e-6

    # Loss Weights
    INSPIRATORY_WEIGHT = 1.0
    EXPIRATORY_WEIGHT = 0.1

    # Early Stopping
    PATIENCE = 20

    # ==========================================
    # Data Processing & Feature Engineering
    # ==========================================
    # Debug flag to limit dataset size for rapid prototyping/debugging
    DEBUG = False

    # Feature Engineering Flags
    USE_ROBUST_SCALER = True
    USE_TIME_WEIGHTED_VOL = True

    # Breath sequence length (approximate max length in dataset is 80)
    MAX_SEQ_LEN = 80
