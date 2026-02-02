import os
import torch


class Config:
    def __init__(self, debug=False, epochs=50, batch_size=32):
        """
        Configuration for the RNA Degradation Prediction Task.

        Args:
            debug (bool): If True, sets parameters for a quick debug run (fewer epochs, smaller batch).
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
        """
        # =========================================================================
        # File Paths and Directories
        # =========================================================================
        self.INPUT_DIR = "./input"
        self.METADATA_DIR = "./metadata"
        self.WORKING_DIR = "./working/idea_71"

        # Ensure working directory exists
        os.makedirs(self.WORKING_DIR, exist_ok=True)

        # Data Paths (using Parquet metadata as generated)
        self.TRAIN_PATH = os.path.join(self.METADATA_DIR, "train.parquet")
        self.VAL_PATH = os.path.join(self.METADATA_DIR, "val.parquet")
        self.TEST_PATH = os.path.join(self.METADATA_DIR, "test.parquet")
        self.SAMPLE_SUBMISSION_PATH = os.path.join(
            self.INPUT_DIR, "sample_submission.csv"
        )

        # Output Paths
        self.SUBMISSION_PATH = os.path.join(self.WORKING_DIR, "submission.csv")
        self.MODEL_PATH = os.path.join(self.WORKING_DIR, "best_model.pth")

        # Cache Directory for deterministic data processing
        self.CACHE_DIR = self.WORKING_DIR

        # =========================================================================
        # Data Specifications
        # =========================================================================
        self.SEQ_LENGTH = 107
        self.SEQ_SCORED = 68
        self.NUM_TARGETS = 5

        # Columns
        self.TARGET_COLS = [
            "reactivity",
            "deg_Mg_pH10",
            "deg_pH10",
            "deg_Mg_50C",
            "deg_50C",
        ]

        # Only these columns are used for the competition metric validation
        self.SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

        # =========================================================================
        # Model Architecture (High-Capacity FFN-Augmented Decoupled BiGRU)
        # =========================================================================
        # Input Channels: 4 (Nucleotide) + 3 (Structure) + 7 (Loop Type) = 14
        self.INPUT_CHANNELS = 14

        # Convolutional Stem
        self.CONV_FILTERS = 256
        self.CONV_KERNEL = 3

        # Backbone
        self.HIDDEN_DIM = 384  # Dimension per direction (Total = 768)
        self.NUM_LAYERS = 4
        self.DROPOUT = 0.1

        # =========================================================================
        # Training Hyperparameters
        # =========================================================================
        self.SEED = 42
        self.LEARNING_RATE = 1e-3
        self.WEIGHT_DECAY = 1e-4
        self.MAX_GRAD_NORM = 1.0
        self.PATIENCE = 15  # Early stopping patience

        # Flexible parameters
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.DEBUG = debug

        # Hardware
        self.NUM_WORKERS = 4
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Debug Overrides
        if self.DEBUG:
            self.EPOCHS = 2
            self.BATCH_SIZE = 8
            self.NUM_WORKERS = 0
            print(f"Debug mode enabled: Epochs={self.EPOCHS}, Batch={self.BATCH_SIZE}")
