import os


class Config:
    """
    Central configuration for the Magnitude-Modulated Hybrid Ensemble (Idea 15).
    Handles paths, hyperparameters, and fixed constants for reproducibility.
    """

    # ==========================================
    # Global Seeding
    # ==========================================
    SEED = 42

    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_15"
    SUBMISSION_DIR = "./submission"

    # Ensure necessary writeable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Data Properties
    # ==========================================
    NUM_SENSORS = 10
    SAMPLING_RATE = 100  # Approx 100Hz (60001 samples in 10 mins)
    SEGMENT_LENGTH = 60001  # Exact number of rows per CSV

    # ==========================================
    # Signal Processing (Spectrograms)
    # ==========================================
    # Single-Resolution Spectrogram generation
    N_FFT = 1024
    HOP_LENGTH = 256  # Results in approx 235 time steps
    IMG_SIZE = (256, 256)  # Target resize dimension for CNN input

    # Normalization Constant for Global Log-Max Scaling
    # Based on int16 range [-32767, 32767] as per dataset description.
    # Formula: X_norm = log(X + 1) / log(GLOBAL_MAX_CONST + 1)
    GLOBAL_MAX_CONST = 32767.0

    # ==========================================
    # Model Architecture
    # ==========================================
    # Branch B: Vision Model
    EFFICIENTNET_VERSION = "efficientnet_b0"
    FILM_HIDDEN_DIM = 64  # Hidden dim for FiLM generator MLP
    FILM_OUTPUT_DIM = 1280  # Matches EfficientNet-B0 final conv channels (usually 1280)

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    N_FOLDS = 5
    BATCH_SIZE = 32
    EPOCHS = 35  # "30+ epochs" requirement
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    PATIENCE = 8  # Early stopping patience

    # ==========================================
    # Tabular Model (LightGBM) Hyperparameters
    # ==========================================
    # Branch A: Energy-Partitioned Tabular Regressor
    LGB_PARAMS = {
        "objective": "regression",
        "metric": "mae",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "n_estimators": 5000,
        "early_stopping_rounds": 100,
        "random_state": SEED,
        "n_jobs": -1,
    }

    # ==========================================
    # Meta-Learner Hyperparameters
    # ==========================================
    META_ALPHA = 1.0  # Ridge regression regularization strength
