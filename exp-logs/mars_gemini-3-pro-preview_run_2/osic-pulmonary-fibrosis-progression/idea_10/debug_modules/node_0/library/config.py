import os


class Config:
    """
    Configuration class for the Spatially-Partitioned Hybrid-Feature Quantile-Elastic Pipeline.
    """

    # ==========================================
    # Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching intermediate files (e.g., features, processed images)
    # Using 'idea_10' as specified in the requirements
    WORKING_DIR = "./working/idea_10"

    # Directory for final submission file
    SUBMISSION_DIR = "./submission"

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # ==========================================
    # Data Preprocessing Constants
    # ==========================================
    # Image resolution for the EfficientNet backbone
    IMG_SIZE = 224

    # Number of anatomical zones to split the lung into (Upper, Middle, Lower)
    N_ZONES = 3

    # Number of bins for the zonal density histogram (Hounsfield Units profiling)
    DENSITY_BINS = 4

    # Hounsfield Unit bins edges could be defined here if fixed,
    # but the logic description suggests generic bins (e.g. <-950, -950:-700, etc.)
    # We will let the feature extractor handle the specific edges,
    # but the count is fixed here.

    # ==========================================
    # Feature Engineering & Model Hyperparameters
    # ==========================================
    # Visual Backbone
    BACKBONE_NAME = "efficientnet_b0"

    # Dimensionality Reduction
    # Mandatory to prevent curse of dimensionality with linear solvers
    PCA_COMPONENTS = 40

    # FVC Predictor (Linear Quantile Regressor)
    # Targeting the Median (L1 loss alignment)
    QUANTILE = 0.5

    # Uncertainty Predictor (Elastic Net)
    # Regularization parameters can be tuned, but defaults are useful
    ELASTIC_L1_RATIO = 0.5

    # ==========================================
    # Metric & Evaluation Constants
    # ==========================================
    # Laplace Log Likelihood constants
    MAX_ERROR = 1000  # Delta is clipped at 1000 ml
    MIN_UNCERTAINTY = 70  # Sigma is clipped at 70 ml

    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    @classmethod
    def setup(cls):
        """
        Creates necessary output directories if they do not exist.
        Should be called at the start of the pipeline.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Automatically setup directories when config is imported
Config.setup()
