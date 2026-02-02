import os
import torch


class Config:
    """
    Configuration for the Hybrid Geometric-Attention Network (HGA-Net) pipeline.
    """

    def __init__(
        self,
        debug: bool = False,
        epochs: int = 50,
        batch_size: int = 128,
        hidden_dim: int = 192,
        num_workers: int = 4,
    ):

        # ==========================================
        # General Settings
        # ==========================================
        self.SEED = 42
        self.DEBUG = debug

        # ==========================================
        # Directories
        # ==========================================
        self.INPUT_DIR = "./input"
        self.METADATA_DIR = "./metadata"
        self.WORKING_DIR = "./working/idea_6"

        # Ensure working directory exists
        os.makedirs(self.WORKING_DIR, exist_ok=True)

        # ==========================================
        # Input File Paths
        # ==========================================
        # Metadata files (Pre-split by molecule)
        self.TRAIN_METADATA = os.path.join(self.METADATA_DIR, "train_metadata.csv")
        self.VAL_METADATA = os.path.join(self.METADATA_DIR, "val_metadata.csv")
        self.TEST_METADATA = os.path.join(self.METADATA_DIR, "test_metadata.csv")

        # Raw Data
        self.STRUCTURES_CSV = os.path.join(self.INPUT_DIR, "structures.csv")
        self.STRUCTURES_DIR = os.path.join(self.INPUT_DIR, "structures")
        self.SAMPLE_SUBMISSION = os.path.join(self.INPUT_DIR, "sample_submission.csv")

        # ==========================================
        # Caching Paths (NPZ format)
        # ==========================================
        self.TRAIN_CACHE = os.path.join(self.WORKING_DIR, "cached_train.npz")
        self.VAL_CACHE = os.path.join(self.WORKING_DIR, "cached_val.npz")
        self.TEST_CACHE = os.path.join(self.WORKING_DIR, "cached_test.npz")

        # ==========================================
        # Model Architecture (HGA-Net)
        # ==========================================
        self.HIDDEN_DIM = hidden_dim

        # Backbone (Directional MPNN)
        self.NUM_MPNN_LAYERS = 4
        self.RBF_SIZE = 50  # Radial Basis Functions
        self.SBF_SIZE = 50  # Spherical Basis Functions
        self.CUTOFF = 5.0  # Geometric cutoff in Angstroms

        # Global Context (Transformer)
        self.NUM_TRANSFORMER_LAYERS = 2
        self.NUM_HEADS = 8

        # Readout
        self.DROPOUT = 0.0  # Deterministic MLP as requested

        # ==========================================
        # Training Hyperparameters
        # ==========================================
        self.EPOCHS = epochs
        self.BATCH_SIZE = batch_size

        # Optimization
        self.LEARNING_RATE = 5e-4
        self.WEIGHT_DECAY = 1e-6
        self.WARMUP_EPOCHS = 3

        # ==========================================
        # System
        # ==========================================
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.NUM_WORKERS = num_workers

        # ==========================================
        # Outputs
        # ==========================================
        self.MODEL_SAVE_PATH = os.path.join(self.WORKING_DIR, "best_model.pt")
        self.SUBMISSION_PATH = os.path.join(self.WORKING_DIR, "submission.csv")

    def __repr__(self):
        """Helper to print configuration."""
        return (
            f"Config(debug={self.DEBUG}, epochs={self.EPOCHS}, "
            f"batch_size={self.BATCH_SIZE}, device={self.DEVICE}, "
            f"hidden_dim={self.HIDDEN_DIM})"
        )
