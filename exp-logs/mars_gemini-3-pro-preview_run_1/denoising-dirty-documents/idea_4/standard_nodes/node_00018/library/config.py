import os
import torch


class Config:
    """
    Centralized configuration for the Denoising task.
    Includes paths, hyperparameters for data, model, training, and ensembles.
    """

    # -------------------------------------------------------------------------
    # File System Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata CSV files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for caching data and saving model checkpoints
    # Using 'idea_4' as the designated workspace for this strategy
    WORKING_DIR = "./working/idea_4"

    # Submission directory and file path
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Hyperparameters
    # -------------------------------------------------------------------------
    # Patch size for random crops during training (160x160)
    PATCH_SIZE = 160

    # Batch size for training
    BATCH_SIZE = 16

    # Number of data loading workers (12 vCPUs available, using 4 is safe)
    NUM_WORKERS = 4

    # -------------------------------------------------------------------------
    # Model Hyperparameters (Deep Supervision U-Net)
    # -------------------------------------------------------------------------
    IN_CHANNELS = 1  # Grayscale input
    OUT_CHANNELS = 1  # Grayscale output
    BASE_FILTERS = 32  # Starting filter count (Standard U-Net size)

    # Enable Deep Supervision (auxiliary heads at intermediate decoder layers)
    DEEP_SUPERVISION = False

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    # Total number of epochs (Cosine Annealing requires full horizon)
    NUM_EPOCHS = 1000

    # Initial learning rate for Adam optimizer
    LEARNING_RATE = 1e-3

    # Random seed for reproducibility
    SEED = 42

    # Compute device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------------------------------
    # Ensemble Strategy
    # -------------------------------------------------------------------------
    # Number of independent models to train for the ensemble
    NUM_MODELS = 1

    # -------------------------------------------------------------------------
    # Development / Debugging
    # -------------------------------------------------------------------------
    # If set to an integer, limits the dataset size for rapid debugging
    MAX_SAMPLES = None

    @classmethod
    def setup(cls):
        """
        Creates necessary output directories if they do not exist.
        Should be called at the start of execution.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
