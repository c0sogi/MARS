import os


class Config:
    """
    Global configuration for the Dual-Moment Axial-Quantile Pipeline.
    """

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42

    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    # Base Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Metadata Paths (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Caching
    # Specific cache directory for this idea iteration to store processed arrays
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_17")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Submission
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Processing Hyperparameters
    # -------------------------------------------------------------------------
    # Image Preprocessing
    IMG_SIZE = 224  # Input size for EfficientNet-B0
    N_SLICES = 5  # Number of axial slices to select based on variance
    BATCH_SIZE = 32  # Batch size for feature extraction

    # Feature Engineering
    PCA_COMPONENTS = 30  # Number of components for PCA dimensionality reduction

    # -------------------------------------------------------------------------
    # Model Hyperparameters
    # -------------------------------------------------------------------------
    # Quantile Regression
    QUANTILES = [0.5]  # Target quantile (Median) for FVC prediction

    # ElasticNet (Uncertainty)
    # Note: Specific alpha/l1_ratio might be tuned, but these are handled in model definition

    # -------------------------------------------------------------------------
    # Metric / Evaluation Constants
    # -------------------------------------------------------------------------
    MIN_CONFIDENCE = 70  # Sigma clipped at 70 ml
    MAX_ERROR = 1000  # Error thresholded at 1000 ml

    # -------------------------------------------------------------------------
    # Compute
    # -------------------------------------------------------------------------
    NUM_WORKERS = 4  # Number of dataloader workers
    DEVICE = "cuda"  # Default device
