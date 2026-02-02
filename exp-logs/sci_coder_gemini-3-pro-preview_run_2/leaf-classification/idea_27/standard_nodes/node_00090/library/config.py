import os


class Config:
    # =========================================================================
    # DIRECTORIES AND PATHS
    # =========================================================================
    # Base directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_27"
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Data files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")

    # Cache files
    CACHE_ZERNIKE_TRAIN = os.path.join(WORKING_DIR, "zernike_train.parquet")
    CACHE_ZERNIKE_VAL = os.path.join(WORKING_DIR, "zernike_val.parquet")
    CACHE_ZERNIKE_TEST = os.path.join(WORKING_DIR, "zernike_test.parquet")

    # Output file
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # GLOBAL CONFIGURATION
    # =========================================================================
    RANDOM_SEED = 42
    N_JOBS = 12  # Utilizing available vCPUs

    # =========================================================================
    # FEATURE ENGINEERING (ZERNIKE MOMENTS)
    # =========================================================================
    # Order of Zernike polynomials to calculate.
    # Order 10 yields approximately 36 features (magnitudes).
    ZERNIKE_ORDER = 10

    # Image processing for moment extraction
    # Images are centered and scaled to radius 1.0, so pixel size is relative.
    # However, loading size affects resolution.
    IMG_LOAD_SIZE = (256, 256)

    # =========================================================================
    # MODEL HYPERPARAMETERS (TIER 1 & TIER 2)
    # =========================================================================

    # Tier 1: Linear Discriminant Analysis (LDA) on Global Features (192 dims)
    # Strategies for shrinkage:
    # - 'ledoit_wolf': sklearn 'auto' (Ledoit-Wolf lemma)
    # - 'oas': Oracle Approximating Shrinkage
    # - float: Fixed shrinkage coefficient
    LDA_SHRINKAGE_STRATEGIES = ["ledoit_wolf", "oas", 0.01, 0.1]

    # Tier 2: Quadratic Discriminant Analysis (QDA) on Zernike Features (~36 dims)
    # Regularization parameters to control covariance estimation
    QDA_REG_PARAMS = [0.0, 0.1, 0.5]

    # Preprocessing for Models
    # 'yeo-johnson' is generally preferred for stabilizing variance
    POWER_TRANSFORM_METHOD = "yeo-johnson"
