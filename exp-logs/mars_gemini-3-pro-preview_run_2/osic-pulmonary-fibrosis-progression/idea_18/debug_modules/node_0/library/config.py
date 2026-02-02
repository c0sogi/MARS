import os


class Config:
    """
    Configuration class for the 2.5D Context-Aware Hybrid Quantile-Elastic Pipeline.
    """

    # =========================================================================
    # PATHS
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Specific Cache Directory for this Idea
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_18")

    # Metadata Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # =========================================================================
    # DATA PREPROCESSING
    # =========================================================================
    # Image Dimensions
    IMG_SIZE = 224

    # 2.5D Slice Selection
    # We select the top N slices with highest variance
    NUM_SLICES = 5

    # Context: +/- 1 slice (z-1, z, z+1) results in 3 channels
    SLICE_CONTEXT = 1
    input_channels = 3

    # Lung Windowing (Hounsfield Units)
    # Used for masking and normalization
    HU_MIN = -1000
    HU_MAX = -400

    # Radiomics
    # Bins for Global Density Histogram: Emphysema, Healthy, Fibrosis, Consolidation
    RADIOMICS_BINS = 4

    # =========================================================================
    # FEATURE ENGINEERING & MODEL
    # =========================================================================
    # Feature Extractor Backbone
    BACKBONE_NAME = "efficientnet_b0"

    # Dimensionality Reduction
    PCA_COMPONENTS = 30

    # =========================================================================
    # TRAINING & OPTIMIZATION
    # =========================================================================
    SEED = 42

    # Quantile for FVC prediction (Median)
    QUANTILE = 0.5

    # Metric Parameters
    CONFIDENCE_CLIP = 70
    MAX_ERROR = 1000

    # Debugging / Development
    # Set to True to run on a small subset of data
    DEBUG = False
    DEBUG_SAMPLES = 20

    # Optimization parameters (if using iterative solvers like SGD/Adam)
    # Although sklearn solvers are used, these serve as placeholders for flexibility
    EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3

    @classmethod
    def setup(cls):
        """
        Creates necessary working and submission directories.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def get_cache_path(cls, identifier, suffix=".npy"):
        """
        Generates a file path for caching processed data.

        Args:
            identifier (str): Unique identifier (e.g., Patient ID).
            suffix (str): File extension.

        Returns:
            str: Full path to the cache file.
        """
        return os.path.join(cls.CACHE_DIR, f"{identifier}{suffix}")
