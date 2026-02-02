import os


class Config:
    """
    Configuration class for the Lung Function Decline Prediction task.
    Implements the settings for the Deep-Feature Varying-Coefficient Elastic Net strategy.
    """

    # ==========================================
    # Global & Reproducibility Settings
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data for debugging

    # ==========================================
    # Directory & File Paths
    # ==========================================
    # Read-only Input
    INPUT_DIR = "./input"
    TRAIN_DICOM_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DICOM_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Working Directories
    WORKING_DIR = "./working"
    # Cache directory for storing intermediate processed features (numpy/parquet)
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_2")

    # Submission Output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Image Processing & Feature Extraction
    # ==========================================
    # Input image size for the CNN backbone (Standard for ResNet/EfficientNet)
    IMG_SIZE = (224, 224)

    # Slice Selection Strategy:
    # We select 3 representative axial slices at 20%, 50%, and 80% of the scan depth
    # to approximate the 3D volume without high computational cost.
    SLICE_SELECTION_RATIOS = [0.2, 0.5, 0.8]

    # CNN Backbone Hyperparameters
    BACKBONE_NAME = "resnet18"
    PRETRAINED = True

    # DataLoader Settings
    BATCH_SIZE = 32
    NUM_WORKERS = 4  # Optimized for 12 vCPUs

    # ==========================================
    # Dimensionality Reduction (PCA)
    # ==========================================
    # Number of principal components to retain from the deep feature vectors
    # Reduces dimensionality before feeding into the Elastic Net
    N_PCA_COMPONENTS = 30

    # ==========================================
    # Model Hyperparameters (Elastic Net)
    # ==========================================
    # Primary Model: Predicts FVC
    # Alpha: Regularization strength
    # L1_Ratio: The ElasticNet mixing parameter (0=Ridge, 1=Lasso)
    ENET_ALPHA = 0.5
    ENET_L1_RATIO = 0.5

    # Secondary Model: Predicts Uncertainty (Sigma)
    # Trained on absolute residuals of the primary model
    SIGMA_ALPHA = 0.5
    SIGMA_L1_RATIO = 0.5

    # ==========================================
    # Metric & Post-Processing
    # ==========================================
    # Constants defined by the Laplace Log Likelihood metric
    MIN_CONFIDENCE = 70.0  # Clipped minimum confidence (ml)
    MAX_ERROR_THRESHOLD = 1000.0  # Error threshold for metric calculation

    @staticmethod
    def setup():
        """
        Creates necessary working directories for caching and submission.
        """
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)


# Ensure directories exist upon import
Config.setup()
