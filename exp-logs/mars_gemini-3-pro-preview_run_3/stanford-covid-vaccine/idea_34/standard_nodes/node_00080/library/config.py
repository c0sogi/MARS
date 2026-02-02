import os
import torch


class Config:
    """
    Configuration class for the Deep Context-Aware Structural-Gated BiGRU (DCASG-BiGRU) strategy.
    Centralizes hyperparameters, file paths, and constants.
    """

    def __init__(self, debug=False):
        # =========================================================================
        # General Settings
        # =========================================================================
        self.PROJECT_NAME = "RNA_Degradation_Prediction"
        self.IDEA_NAME = "idea_34"
        self.SEED = 42
        self.DEBUG = debug

        # =========================================================================
        # Paths
        # =========================================================================
        self.INPUT_DIR = "./input"
        self.METADATA_DIR = "./metadata"
        self.WORKING_DIR = f"./working/{self.IDEA_NAME}"

        # Ensure working directory exists
        os.makedirs(self.WORKING_DIR, exist_ok=True)

        # Raw Input Files
        self.TRAIN_JSON = os.path.join(self.INPUT_DIR, "train.json")
        self.TEST_JSON = os.path.join(self.INPUT_DIR, "test.json")
        self.SAMPLE_SUBMISSION = os.path.join(self.INPUT_DIR, "sample_submission.csv")

        # Metadata Files (Parquet)
        self.TRAIN_PARQUET = os.path.join(self.METADATA_DIR, "train.parquet")
        self.VAL_PARQUET = os.path.join(self.METADATA_DIR, "val.parquet")
        self.TEST_PARQUET = os.path.join(self.METADATA_DIR, "test.parquet")

        # Cache Files (Numpy/Pickle alternative)
        # Using .npz for efficient array storage
        self.TRAIN_CACHE = os.path.join(self.WORKING_DIR, "train_data_cache.npz")
        self.VAL_CACHE = os.path.join(self.WORKING_DIR, "val_data_cache.npz")
        self.TEST_CACHE = os.path.join(self.WORKING_DIR, "test_data_cache.npz")

        # Output Artifacts
        self.BEST_MODEL_PATH = os.path.join(self.WORKING_DIR, "best_model.pth")
        self.SUBMISSION_PATH = os.path.join(self.WORKING_DIR, "submission.csv")

        # =========================================================================
        # Data Dimensions & Features
        # =========================================================================
        self.SEQ_LEN = 107
        self.SEQ_SCORED = 68

        # Input Channels:
        # Sequence (4: A,G,U,C) + Structure (3: (,.,)) + Loop Type (7: S,M,I,B,H,E,X)
        self.INPUT_DIM = 4 + 3 + 7  # 14
        self.NUM_TARGETS = 5

        # Feature Mappings
        self.TOKEN_MAP_SEQ = {"A": 0, "G": 1, "U": 2, "C": 3}
        self.TOKEN_MAP_STRUCT = {"(": 0, ")": 1, ".": 2}
        self.TOKEN_MAP_LOOP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

        # =========================================================================
        # Model Hyperparameters (DCASG-BiGRU)
        # =========================================================================
        self.HIDDEN_DIM = 384  # Width for capacity
        self.NUM_LAYERS = 4  # Depth for complex interactions
        self.KERNEL_SIZE = 3  # For Convolutional Stem
        self.DROPOUT = 0.1
        self.CNN_FILTERS = 256  # For Convolutional Stem

        # =========================================================================
        # Training Hyperparameters
        # =========================================================================
        self.BATCH_SIZE = 16
        self.EPOCHS = 20 if not self.DEBUG else 2
        self.LEARNING_RATE = 1e-3
        self.WEIGHT_DECAY = 1e-4
        self.MAX_GRAD_NORM = 1.0  # Gradient Clipping
        self.PATIENCE = 5  # Early Stopping

        # Scheduler Settings (Cosine Annealing)
        self.T_MAX = self.EPOCHS
        self.ETA_MIN = 1e-6

        # =========================================================================
        # Evaluation & Scoring
        # =========================================================================
        # All targets present in training data
        self.TARGET_COLS = [
            "reactivity",
            "deg_Mg_pH10",
            "deg_pH10",
            "deg_Mg_50C",
            "deg_50C",
        ]

        # Targets used for the competition metric
        self.SCORING_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

        # Indices of scoring cols within target cols
        self.SCORING_INDICES = [0, 1, 3]

        # =========================================================================
        # Hardware
        # =========================================================================
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.NUM_WORKERS = 4


# Instantiate a default config object for easy import
config = Config()
