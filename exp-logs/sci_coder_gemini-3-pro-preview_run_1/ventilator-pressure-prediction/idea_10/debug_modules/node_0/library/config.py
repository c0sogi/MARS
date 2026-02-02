import os
import torch


class Config:
    """
    Configuration for Deeply Supervised Physics-Injected Residual Multi-Scale CNN-LSTM.
    Acts as the single source of truth for paths, hyperparameters, and feature definitions.
    """

    # ==========================================
    # Path Configuration
    # ==========================================
    METADATA_DIR = "./metadata"
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for artifacts (cache, models, submissions)
    WORKING_DIR = "./working/idea_10"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    MODEL_PATH = os.path.join(WORKING_DIR, "model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # Ensure working directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)

    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # Model Architecture Hyperparameters
    # ==========================================
    # Backbone: 4-layer Bidirectional LSTM with Context Injection and Residuals
    HIDDEN_SIZE = 512
    NUM_LAYERS = 4
    DROPOUT = 0.1
    BIDIRECTIONAL = True

    # Multi-Scale Stem (Inception-1D) Kernels
    CNN_KERNELS = [3, 5, 7]

    # Deep Supervision: Weight for the auxiliary regression head at Layer 2
    AUX_WEIGHT = 0.3

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    EPOCHS = 35
    BATCH_SIZE = 512

    # Optimizer (AdamW) & Scheduler (OneCycleLR)
    LR = 1e-3  # Max learning rate
    WEIGHT_DECAY = 1e-2
    PCT_START = 0.3  # Percentage of training to increase LR
    DIV_FACTOR = 25.0  # Initial LR = LR / DIV_FACTOR
    FINAL_DIV_FACTOR = 1000.0
    CLIP_GRAD = 1.0  # Gradient clipping value

    # ==========================================
    # Data & Feature Engineering
    # ==========================================
    SEQ_LEN = 80
    TARGET_COL = "pressure"
    ID_COL = "id"
    BREATH_ID_COL = "breath_id"

    # Raw columns expected in the input CSVs
    RAW_COLS = ["id", "breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]

    # Complete list of features to be used as input to the model.
    # The preprocessing pipeline must generate these features in this exact order.
    FEATURE_COLS = [
        "time_step",
        "u_in",
        "u_out",
        "R",  # Continuous scaled
        "C",  # Continuous scaled
        "u_in_cumsum",  # Proxy for Volume (Integral of u_in)
        "u_in_lag1",  # Lag features
        "u_in_lag2",
        "u_in_lag3",
        "u_in_lag4",
        "u_in_diff1",  # First difference
        "u_in_diff2",  # Second difference
        "R_u_in",  # Interaction: R * u_in (Pressure drop ~ Flow * Resistance)
        "vol_C",  # Interaction: Volume / C (Pressure ~ Volume / Compliance)
    ]

    # Subset of features to be injected into the LSTM at every time step (Deep Context Injection).
    # These represent the physical constraints and key state variables.
    CONTEXT_FEATURES = ["R", "C", "R_u_in", "vol_C"]

    # Dimensions derived from feature lists
    INPUT_DIM = len(FEATURE_COLS)
    CONTEXT_DIM = len(CONTEXT_FEATURES)

    # ==========================================
    # Runtime / Hardware
    # ==========================================
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 2000  # Number of breaths to use when DEBUG=True

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
