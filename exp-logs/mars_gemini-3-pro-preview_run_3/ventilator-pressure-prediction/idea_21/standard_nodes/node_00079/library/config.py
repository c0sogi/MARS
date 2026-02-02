import os
import torch


class Config:
    """
    Configuration for the Residual-Dense Hybrid Network (RDH-Net) pipeline.
    Defines hyperparameters, file paths, and feature sets.
    """

    # ==========================================
    # General Configuration
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to use a subset of data for debugging
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    # Strictly use Batch Size 128 as per "Stabilized Critical Mass Regime"
    BATCH_SIZE = 128
    # Extended training budget for hybrid convergence
    EPOCHS = 80

    # Optimizer settings
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    # Mandatory gradient clipping
    MAX_GRAD_NORM = 1.0

    # Scheduler settings
    WARMUP_EPOCHS = 5
    COSINE_CYCLES = 0.5

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 15

    # ==========================================
    # Model Architecture (RDH-Net)
    # ==========================================
    # Branch 1: Deep Residual Dense TCN (Resistive Stream)
    # Pyramidal scaling to match LSTM capacity (Cite 00057, 00024)
    CNN_CHANNELS = [64, 128, 256, 512, 512, 512]
    KERNEL_SIZE = 9
    CNN_DROPOUT = 0.1

    # Branch 2: High-Capacity Bi-LSTM (Elastic Stream)
    LSTM_INPUT_DIM = None  # Will be set based on len(INPUT_FEATURES)
    LSTM_HIDDEN = 512
    LSTM_LAYERS = 3
    LSTM_BIDIRECTIONAL = True
    LSTM_DROPOUT = 0.1

    # Fusion Head: Wide-Latent Integration
    FUSION_HIDDEN = 1024

    # ==========================================
    # Data Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_21"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = "./submission"

    # Metadata paths (Pre-split data)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Raw input paths
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output paths
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SCALER_PATH = os.path.join(WORKING_DIR, "scaler.joblib")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Feature Engineering & Selection
    # ==========================================
    # Columns to load from CSV
    LOAD_COLS = ["id", "breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
    TEST_LOAD_COLS = ["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]

    # Target variable
    TARGET_COL = "pressure"
    ID_COL = "id"
    BREATH_ID_COL = "breath_id"

    # 1. Dynamic State
    # We explicitly retain u_out to allow Bi-LSTM to explain future expiratory states
    DYNAMIC_FEATURES = ["u_in", "u_out"]

    # 2. Physically Accurate State (Engineered)
    # dt: Time delta (variable rate physics)
    # u_in_diff: Acceleration/Derivative
    # area: Integral volume (sum u_in * dt)
    ENGINEERED_FEATURES = ["dt", "u_in_diff", "area"]

    # 3. Static Physics & Interactions
    # R, C: Lung attributes
    # R_u_in: Resistive pressure component proxy
    # area_C: Elastic pressure component proxy
    STATIC_AND_INTERACTION_FEATURES = ["R", "C", "R_u_in", "area_C"]

    # 4. Lookahead Features
    # Explicitly shifted columns for zero-lag future context
    LOOKAHEAD_STEPS = 4
    LOOKAHEAD_FEATURES = [f"u_in_next_{i}" for i in range(1, LOOKAHEAD_STEPS + 1)]

    # Final Input Feature List
    # Raw 'time_step' is EXCLUDED to prevent translation-variance instability
    INPUT_FEATURES = (
        DYNAMIC_FEATURES
        + ENGINEERED_FEATURES
        + STATIC_AND_INTERACTION_FEATURES
        + LOOKAHEAD_FEATURES
    )

    @classmethod
    def setup(cls):
        """Creates necessary directories for the pipeline."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def get_input_dim(cls):
        """Returns the number of input features."""
        return len(cls.INPUT_FEATURES)
