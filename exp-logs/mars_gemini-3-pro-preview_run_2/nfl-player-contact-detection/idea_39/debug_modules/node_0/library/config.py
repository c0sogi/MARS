import os
import torch


class Config:
    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching intermediate files (Parquet, Scalers, etc.)
    WORKING_DIR = "./working/idea_39"

    # Directory for final submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure necessary writeable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Reproducibility & Compute
    # =========================================================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Optimized for 12 vCPUs

    # =========================================================================
    # Data Processing / Feature Engineering
    # =========================================================================
    # Temporal Window: t-5 to t+5 (Current step + 5 past + 5 future)
    WINDOW_PRE = 5
    WINDOW_POST = 5

    # Feature Clamping Ranges (Physical Stability)
    # Used during pre-processing to remove sensor outliers
    PHYSICAL_RANGES = {
        "speed": 35.0,  # Yards/sec
        "acceleration": 40.0,  # Yards/sec^2
        "distance": 100.0,  # Yards
        "orientation": 360.0,  # Degrees
        "direction": 360.0,  # Degrees
    }

    # Input Layer Clamping (Model-side)
    # Global clamp for the input tensor to ensure gradients don't explode
    INPUT_CLAMP_MIN = -50.0
    INPUT_CLAMP_MAX = 50.0

    # Caching Filenames
    CACHE_TRAIN_PARQUET = os.path.join(WORKING_DIR, "train_features.parquet")
    CACHE_VAL_PARQUET = os.path.join(WORKING_DIR, "val_features.parquet")
    CACHE_TEST_PARQUET = os.path.join(WORKING_DIR, "test_features.parquet")
    CACHE_SCALER = os.path.join(WORKING_DIR, "scaler.joblib")

    # =========================================================================
    # Model Architecture: WD-PIRV
    # =========================================================================
    # Pyramidal Backbone Dimensions: Input -> 512 -> 256 -> 128 -> Logit
    PYRAMIDAL_DIMS = [512, 256, 128]

    # Visual Branch Dimensions
    VISUAL_HIDDEN_DIM = 64

    # Fusion Parameter
    # Initial weight for visual branch (lambda) can be learned or fixed
    VISUAL_LAMBDA_INIT = 0.5

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 8192
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Training Duration
    EPOCHS = 20
    EARLY_STOPPING_PATIENCE = 3

    # Focal Loss Parameters
    # Alpha balances positive/negative class importance
    # Gamma focuses on hard examples
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0

    # Regularization
    # Gaussian Noise Sigma for input injection during training
    NOISE_SIGMA = 0.05
    DROPOUT_RATE = 0.1

    # =========================================================================
    # Model Artifacts
    # =========================================================================
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    THRESHOLD_PATH = os.path.join(WORKING_DIR, "best_threshold.npy")
