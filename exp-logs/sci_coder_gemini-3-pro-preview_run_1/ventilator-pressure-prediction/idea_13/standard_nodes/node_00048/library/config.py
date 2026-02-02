import os
import torch


class Config:
    """
    Configuration class for the Ventilator Pressure Prediction task.
    Implements settings for the High-Capacity Unnormalized Physics-Injected
    Composite CNN-LSTM-FFN strategy.
    """

    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata splits (guaranteed disjoint breath_ids)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for caching engineered features and model checkpoints
    WORKING_DIR = "./working/idea_13"
    CACHE_DIR = WORKING_DIR

    # Model checkpoint path
    MODEL_PATH = os.path.join(WORKING_DIR, "model.pth")

    # Submission output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Reproducibility
    # =========================================================================
    SEED = 42

    # =========================================================================
    # Data Pipeline Hyperparameters
    # =========================================================================
    # Time series length per breath
    SEQ_LEN = 80

    # Feature Engineering Configuration
    # These lists define what columns will be generated and used.

    # Base columns provided in raw data
    raw_features = ["time_step", "u_in", "u_out", "R", "C"]

    # Features to be used as input to the model
    # Note: 'u_out' is binary and used for masking, but also included as a feature.
    # The 'pressure' column is the target.

    # Continuous features that will be Scaled (RobustScaler)
    SCALABLE_FEATURES = [
        "time_step",
        "u_in",
        "R",
        "C",
        "area",  # Volume approximation (cumsum(u_in * dt))
        "u_in_cumsum",  # Raw cumsum of u_in
        "u_in_lag1",
        "u_in_lag2",
        "u_in_lag3",
        "u_in_lag4",
        "u_in_diff1",
        "u_in_diff2",
        "R_u_in",  # Interaction: R * u_in (Physics injection)
        "area_div_C",  # Interaction: Volume / C (Physics injection)
    ]

    # All input features (Scalable + Binary/Categorical if any)
    # u_out is binary, we include it directly.
    INPUT_FEATURES = SCALABLE_FEATURES + ["u_out"]

    # Input dimension for the model
    INPUT_DIM = len(INPUT_FEATURES)

    # Data Loading
    NUM_WORKERS = 4

    # =========================================================================
    # Model Architecture Hyperparameters
    # =========================================================================
    # Strategy: High-Capacity Unnormalized Composite Block
    HIDDEN_SIZE = 512

    # Number of Composite Blocks (LSTM + FFN)
    NUM_LAYERS = 4

    # Stem CNN Kernel Sizes for Multi-Scale extraction
    CNN_KERNEL_SIZES = [3, 5, 7]

    # Dropout rate applied to residual branches
    DROPOUT = 0.1

    # Normalization: Explicitly False to preserve pressure magnitude
    USE_LAYER_NORM = False

    # Bidirectional LSTM
    BIDIRECTIONAL = True

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    # Extended optimization horizon for OneCycleLR annealing
    EPOCHS = 35

    # Fixed batch size to ensure sufficient gradient updates
    BATCH_SIZE = 512

    # Optimizer settings
    LR_MAX = 1e-3
    WEIGHT_DECAY = 1e-2

    # Loss weighting
    # Total Loss = L1_Masked(Main) + AUX_WEIGHT * L1_Masked(Aux)
    AUX_WEIGHT = 0.3

    # Device configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def print_config():
        """Prints the current configuration."""
        print("=" * 40)
        print("CONFIG")
        print("=" * 40)
        print(f"Model: High-Capacity Unnormalized Composite CNN-LSTM-FFN")
        print(f"Hidden Size: {Config.HIDDEN_SIZE}")
        print(f"Layers: {Config.NUM_LAYERS}")
        print(f"Epochs: {Config.EPOCHS}")
        print(f"Batch Size: {Config.BATCH_SIZE}")
        print(f"Input Features: {len(Config.INPUT_FEATURES)}")
        print(f"Device: {Config.DEVICE}")
        print(f"Cache Dir: {Config.CACHE_DIR}")
        print("=" * 40)
