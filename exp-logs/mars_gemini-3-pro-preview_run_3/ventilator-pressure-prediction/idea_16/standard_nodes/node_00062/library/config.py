import os
import torch


class Config:
    """
    Global configuration for the Pyramidal Inception-TCN Hybrid (PITH-Net) pipeline.
    Synchronizes feature engineering, model architecture, and training hyperparameters.
    """

    # ==============================
    # General Configuration
    # ==============================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging

    # ==============================
    # Paths
    # ==============================
    # Input Metadata (generated in ./metadata)
    TRAIN_PATH = "./metadata/train.csv"
    VAL_PATH = "./metadata/validation.csv"
    TEST_PATH = "./metadata/test.csv"

    # Working Directory for Caching
    # Stores processed numpy arrays to speed up subsequent runs
    WORKING_DIR = "./working/idea_16/"

    # Output Directory for Submission
    SUBMISSION_DIR = "./submission/"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==============================
    # Feature Engineering Config
    # ==============================
    # Column names in the raw dataset
    ID_COL = "id"
    BREATH_ID_COL = "breath_id"
    TIME_COL = "time_step"
    TARGET_COL = "pressure"

    # Complete list of features to be used by the model.
    # The data processing pipeline must generate these exact columns.
    FEATURE_COLS = [
        # --- Raw Control & State ---
        "time_step",  # Temporal Context
        "u_in",  # Inspiratory Valve Control
        "u_out",  # Expiratory Valve Control
        "R",  # Lung Resistance
        "C",  # Lung Compliance
        # --- Dynamic Physics (Derived) ---
        "area",  # Physically Accurate Volume (Integral of u_in)
        "u_in_diff",  # Acceleration (Derivative of u_in)
        # --- Lookahead Injection (Future Context) ---
        "u_in_next1",  # u_in at t+1
        "u_in_next2",  # u_in at t+2
        "u_in_next3",  # u_in at t+3
        "u_in_next4",  # u_in at t+4
        "u_in_diff_next1",  # Derivative at t+1
        # --- Static Physics Interactions ---
        "R_u_in",  # Interaction: Resistance * Flow
        "area_C",  # Interaction: Volume / Compliance
    ]

    # Input dimension for the neural network
    INPUT_DIM = len(FEATURE_COLS)

    # ==============================
    # Model Hyperparameters (PITH-Net)
    # ==============================
    # Branch 1: Pyramidal Inception-TCN (Resistive Stream)
    # Inception module uses multiple kernel sizes in parallel
    TCN_KERNEL_SIZES = [3, 7, 11]
    # Channel capacity doubles at each layer (Pyramidal)
    TCN_CHANNELS = [64, 128, 256, 512]
    TCN_DROPOUT = 0.1

    # Branch 2: High-Capacity Bidirectional LSTM (Elastic Stream)
    LSTM_HIDDEN_SIZE = 512
    LSTM_LAYERS = 3
    LSTM_BIDIRECTIONAL = True

    # ==============================
    # Training Hyperparameters
    # ==============================
    # "Small Batch Size" strategy to introduce gradient noise
    BATCH_SIZE = 128

    # "Extended Training" strategy for hybrid convergence
    EPOCHS = 80

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler (ReduceLROnPlateau)
    SCHEDULER_PATIENCE = 5
    SCHEDULER_FACTOR = 0.5
    MIN_LR = 1e-6

    # Hardware
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Ensures necessary directories exist.
        Should be called at the start of the pipeline.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Automatically setup directories when config is imported
Config.setup()
