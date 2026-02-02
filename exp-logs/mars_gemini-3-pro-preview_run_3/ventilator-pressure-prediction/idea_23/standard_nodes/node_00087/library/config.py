import os
import torch


class Config:
    """
    Configuration for the Residual-Dense Hybrid Network (RDH-Net) experiment.
    Defines hyperparameters, file paths, and feature engineering specifications.
    """

    def __init__(self, debug: bool = False):
        # Reproducibility
        self.SEED = 42

        # Execution Mode
        self.DEBUG = debug
        # In debug mode, we use a small subset of breaths and fewer epochs
        self.DEBUG_BREATH_COUNT = 100 if debug else None

        # Paths
        self.INPUT_DIR = "./input"
        self.METADATA_DIR = "./metadata"
        self.WORKING_DIR = "./working/idea_23"
        self.SUBMISSION_DIR = "./submission"

        # Metadata Files (Pre-split)
        self.TRAIN_PATH = os.path.join(self.METADATA_DIR, "train.csv")
        self.VAL_PATH = os.path.join(self.METADATA_DIR, "validation.csv")
        self.TEST_PATH = os.path.join(self.METADATA_DIR, "test.csv")
        self.SAMPLE_SUBMISSION_PATH = os.path.join(
            self.INPUT_DIR, "sample_submission.csv"
        )

        # Ensure working directories exist
        os.makedirs(self.WORKING_DIR, exist_ok=True)
        os.makedirs(self.SUBMISSION_DIR, exist_ok=True)

        # Caching & Preprocessing
        self.CACHE_TRAIN = os.path.join(self.WORKING_DIR, "train_cache.parquet")
        self.CACHE_VAL = os.path.join(self.WORKING_DIR, "val_cache.parquet")
        self.CACHE_TEST = os.path.join(self.WORKING_DIR, "test_cache.parquet")
        self.SCALER_PATH = os.path.join(self.WORKING_DIR, "scaler.joblib")

        # Feature Engineering Configuration
        self.NUM_LAGS = 4  # Number of future time steps (lookahead) for u_in

        # Base features + Physics features
        # Note: 'time_step' is excluded from input as per RDH-Net strategy
        # 'pressure' is the target and not an input feature
        self.FEATURE_COLS = [
            "u_in",
            "u_out",
            "R",
            "C",
            "dt",  # Time delta
            "area",  # Numerical integral of u_in * dt
            "R__u_in",  # Interaction: R * u_in
            "area__C",  # Interaction: area / C
        ]

        # Add lookahead features dynamically
        for i in range(1, self.NUM_LAGS + 1):
            self.FEATURE_COLS.append(f"u_in_next{i}")

        self.INPUT_DIM = len(self.FEATURE_COLS)

        # Model Architecture: RDH-Net
        # Branch 1: Residual Dense TCN (Resistive Stream)
        self.CONV_KERNEL_SIZE = 9
        self.CONV_FILTERS = [64, 128, 256, 512]  # Progressive channel scaling

        # Branch 2: Bi-LSTM (Elastic Stream)
        self.LSTM_HIDDEN_SIZE = 512
        self.LSTM_LAYERS = 3
        self.LSTM_BIDIRECTIONAL = True

        # Fusion Head
        self.DENSE_HIDDEN_SIZE = 1024
        self.DROPOUT = 0.1

        # Training Hyperparameters
        self.BATCH_SIZE = 128 if not self.DEBUG else 16
        self.EPOCHS = 80 if not self.DEBUG else 2
        self.LEARNING_RATE = 1e-3
        self.WEIGHT_DECAY = 1e-2
        self.CLIP_GRAD_NORM = 1.0  # Strict requirement for hybrid stability

        # Optimization Strategy
        self.SCHEDULER_PATIENCE = 5
        self.SCHEDULER_FACTOR = 0.5
        self.EARLY_STOPPING_PATIENCE = 15

        # Hardware
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.NUM_WORKERS = 4

    def display(self):
        """Prints the configuration."""
        print("=" * 30)
        print(f"Config: RDH-Net (Debug={self.DEBUG})")
        print(f"Device: {self.DEVICE}")
        print(f"Batch Size: {self.BATCH_SIZE}")
        print(f"Epochs: {self.EPOCHS}")
        print(f"Input Features: {len(self.FEATURE_COLS)}")
        print(f"Working Dir: {self.WORKING_DIR}")
        print("=" * 30)
