import os
import torch


class Config:
    """
    Configuration class for the RNA Degradation Prediction pipeline.
    Implements the 'Spatially-Augmented Convolutional BiGRU' strategy settings.
    """

    def __init__(self, debug=False, epochs=50, batch_size=64):
        # =================================================================
        # Environment & Reproducibility
        # =================================================================
        self.SEED = 42
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.NUM_WORKERS = 4  # Adjust based on available vCPUs (12 available)
        self.DEBUG = debug

        # =================================================================
        # Paths
        # =================================================================
        self.INPUT_DIR = "./input"
        self.METADATA_DIR = "./metadata"
        self.WORKING_DIR = "./working/idea_8"

        # Create working directory if it doesn't exist
        os.makedirs(self.WORKING_DIR, exist_ok=True)

        # Data Files
        self.TRAIN_PATH = os.path.join(self.METADATA_DIR, "train.parquet")
        self.VAL_PATH = os.path.join(self.METADATA_DIR, "val.parquet")
        self.TEST_PATH = os.path.join(self.METADATA_DIR, "test.parquet")
        self.SAMPLE_SUBMISSION = os.path.join(self.INPUT_DIR, "sample_submission.csv")

        # Output Files
        self.SUBMISSION_PATH = os.path.join(self.WORKING_DIR, "submission.csv")
        self.MODEL_PATH = os.path.join(self.WORKING_DIR, "best_model.pth")

        # Cache Paths (for processed tensors)
        self.TRAIN_CACHE = os.path.join(self.WORKING_DIR, "train_data.pt")
        self.VAL_CACHE = os.path.join(self.WORKING_DIR, "val_data.pt")
        self.TEST_CACHE = os.path.join(self.WORKING_DIR, "test_data.pt")

        # =================================================================
        # Data Specifications
        # =================================================================
        self.SEQ_LEN = 107
        self.PRED_LEN = 68

        # Feature Dimensions
        # Sequence (4: A,G,U,C) + Structure (3: (, ), .) + Loop (7: S,M,I,B,H,E,X)
        self.BASE_FEATURE_DIM = 14
        # Spatial Augmentation: Concatenating features of paired bases (or zero if unpaired)
        # Input Dim = Base Features (14) + Paired Base Features (14) = 28
        self.INPUT_DIM = 28

        # Target Columns
        self.TARGET_COLS = [
            "reactivity",
            "deg_Mg_pH10",
            "deg_pH10",
            "deg_Mg_50C",
            "deg_50C",
        ]
        self.NUM_CLASSES = len(self.TARGET_COLS)

        # Columns used for the official metric calculation
        self.SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

        # Mappings (One-Hot Encoding Indices)
        self.TOKEN_DICT_SEQ = {x: i for i, x in enumerate("AGUC")}
        self.TOKEN_DICT_STRUCT = {x: i for i, x in enumerate("().")}
        self.TOKEN_DICT_LOOP = {x: i for i, x in enumerate("SMIBHEX")}

        # =================================================================
        # Model Hyperparameters
        # =================================================================
        # Convolutional Stem
        self.CONV_FILTERS = 256
        self.KERNEL_SIZE = 3

        # Recurrent Backbone (BiGRU)
        self.HIDDEN_DIM = 256
        self.NUM_LAYERS = 2
        self.DROPOUT = 0.3

        # =================================================================
        # Training Hyperparameters
        # =================================================================
        self.BATCH_SIZE = batch_size
        self.LEARNING_RATE = 1e-3
        self.EPOCHS = epochs
        self.PATIENCE = 10  # Early stopping patience
        self.WEIGHT_DECAY = 1e-4

        # Loss Function
        # We use unweighted MCRMSE as per strategy
        self.LOSS_FN = "MCRMSE"

    def __repr__(self):
        return (
            f"Config(debug={self.DEBUG}, epochs={self.EPOCHS}, "
            f"batch_size={self.BATCH_SIZE}, device={self.DEVICE})"
        )
