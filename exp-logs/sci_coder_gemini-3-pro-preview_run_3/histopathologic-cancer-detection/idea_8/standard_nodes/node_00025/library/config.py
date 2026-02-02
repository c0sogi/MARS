import os
import torch


class Config:
    """
    Configuration for the Converged Heterogeneous Stacking Ensemble.
    Defines hyperparameters, file paths, and compute settings.
    """

    # --- Global Random Seed ---
    SEED = 42

    # --- Compute Environment ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Using 8 workers to balance I/O with the 12 vCPUs available
    NUM_WORKERS = 8

    # --- File Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_8"
    SUBMISSION_DIR = "./submission"

    # Metadata Files
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Model Configuration ---
    # Heterogeneous backbones for stacking
    MODEL_ARCHS = ["convnext_tiny"]
    NUM_CLASSES = 1

    # --- Data Preprocessing ---
    # Center crop to 64x64 to capture 32x32 ROI + 16px context
    CROP_SIZE = 64

    # --- Training Hyperparameters ---
    NUM_FOLDS = 5
    EPOCHS = 30  # Extended to 30 to ensure convergence based on previous analysis

    # Batch size optimized for A100 40GB VRAM and small 64x64 images
    # Large batch size stabilizes BN statistics and improves throughput
    BATCH_SIZE = 2048

    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-4

    # Cosine Annealing parameters
    ETA_MIN = 1e-6

    # --- Meta-Learner Configuration ---
    # Number of TTA views for OOF generation and Inference
    TTA_STEPS = 4

    # --- Debugging & Development ---
    # Set DEBUG to True to run on a small subset of data
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 5000

    @classmethod
    def setup(cls):
        """
        Initialize the working environment.
        Ensures that necessary directories for checkpoints and submissions exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
