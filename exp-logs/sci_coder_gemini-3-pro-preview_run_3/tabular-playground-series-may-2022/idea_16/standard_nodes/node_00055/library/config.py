import os
import torch


class Config:
    """
    Configuration class for the Noise-Regularized Funnel MLP strategy.
    Centralizes all hyperparameters, file paths, and execution settings.
    """

    # --------------------------------------------------------------------------
    # General Reproducibility & Hardware
    # --------------------------------------------------------------------------
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # 12 vCPUs available, 4 workers is usually a safe optimal point for dataloaders
    NUM_WORKERS = 4

    # --------------------------------------------------------------------------
    # File Paths
    # --------------------------------------------------------------------------
    # Input Data (using generated metadata for correct splits)
    TRAIN_DATA_PATH = "./metadata/train.csv"
    VAL_DATA_PATH = "./metadata/val.csv"
    TEST_DATA_PATH = "./metadata/test.csv"
    SAMPLE_SUBMISSION_PATH = "./input/sample_submission.csv"

    # Working Directory (for checkpoints and cache)
    WORKING_DIR = "./working/idea_16"

    # Submission Directory
    SUBMISSION_DIR = "./submission"

    # Specific Output Files
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Files for Deterministic Processing
    # Using .parquet for dataframes and .npy for metadata/vocab sizes
    TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_processed.parquet")
    VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_processed.parquet")
    TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_processed.parquet")
    METADATA_CACHE_PATH = os.path.join(WORKING_DIR, "metadata.npy")

    # --------------------------------------------------------------------------
    # Model Architecture Hyperparameters
    # --------------------------------------------------------------------------
    # Entity Embeddings
    EMBEDDING_DIM = 16  # Optimal capacity avoiding bottleneck (8) and noise (32)

    # Funnel Backbone
    HIDDEN_LAYERS = [512, 256, 128]  # Compressive inductive bias
    DROPOUT_RATE = 0.2

    # Regularization Innovation
    NOISE_SIGMA = 0.1  # Standard deviation for Gaussian Noise Injection Layer

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 1024  # Moderate batch size for implicit gradient noise regularization
    EPOCHS = 20  # Sufficient for super-convergence with OneCycle
    LEARNING_RATE = 1e-3  # Max LR for OneCycle Policy
    WEIGHT_DECAY = 1e-5  # Calibrated weight decay (less aggressive than default 1e-2)

    @classmethod
    def create_directories(cls):
        """
        Ensures that the necessary working and submission directories exist.
        Should be called at the start of the pipeline.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
