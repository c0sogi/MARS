import os
import torch


class Config:
    """
    Configuration class for the Parsimonious Beamformed Stacking Strategy.
    Defines global constants, file paths, and hyperparameters for all model branches.
    """

    # --------------------------------------------------------------------------
    # Global & Reproducibility
    # --------------------------------------------------------------------------
    SEED = 42
    N_FOLDS = 5
    NUM_WORKERS = 4  # Number of DataLoader workers, adjusted for 12 vCPUs
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debug Flag: Set to True to run on a small subset (e.g., 200 samples)
    # Useful for testing the pipeline within the time limit
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 200

    # --------------------------------------------------------------------------
    # Directory Structure & Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    TRAIN_DATA_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DATA_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Working Directory for Caching (Idea 9)
    # Stores intermediate processed features (parquet/npy) and model checkpoints
    WORKING_DIR = "./working/idea_9"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Submission Output
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Specifications
    # --------------------------------------------------------------------------
    SENSOR_COLS = [f"sensor_{i}" for i in range(1, 11)]
    NUM_SENSORS = 10

    # Signal Properties
    # 10 minutes of data per segment -> 600 seconds
    # 60,001 data points -> Sampling Rate approx 100 Hz
    SAMPLE_RATE = 100
    SIGNAL_LENGTH = 60001

    # --------------------------------------------------------------------------
    # Feature Extraction Configuration
    # --------------------------------------------------------------------------

    # Branch A: Tabular (MFCCs + Beamforming)
    # Parsimonious set: Coefficients 1-13 only to avoid high-freq noise
    N_MFCC = 13

    # Branch B: Vision (Spectrograms)
    # Parameters to generate Log-Mel Spectrograms
    N_FFT = 1024
    HOP_LENGTH = 256
    N_MELS = 64
    F_MIN = 0
    F_MAX = None

    # CNN Input Image Size (Height, Width)
    # Height = N_MELS (64)
    # Width = (60001 / 256) + 1 ≈ 235
    # We resize to a standard square for EfficientNet
    IMG_SIZE = (224, 224)

    # --------------------------------------------------------------------------
    # Model Hyperparameters
    # --------------------------------------------------------------------------

    # Branch A: LightGBM
    LGB_PARAMS = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "n_estimators": 10000,  # Controlled by early stopping
        "early_stopping_rounds": 100,
        "verbose": -1,
        "random_state": SEED,
        "n_jobs": -1,
    }

    # Branch B: EfficientNet-B0
    CNN_MODEL_NAME = "efficientnet_b0"
    CNN_PARAMS = {
        "in_channels": 10,  # 10 stacked sensor channels
        "num_classes": 1,
        "dropout": 0.2,
        "pretrained": True,
    }

    CNN_TRAIN_PARAMS = {
        "batch_size": 32,
        "epochs": 35,  # Extended duration (>30) for convergence
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "eta_min": 1e-6,  # For Cosine Annealing
        "patience": 5,
        "num_workers": NUM_WORKERS,
    }

    # Meta-Learner: Ridge Regression
    # Unconstrained stacking to allow bias correction
    RIDGE_ALPHA = 1.0
