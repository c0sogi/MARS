import os
import numpy as np

# =============================================================================
# GLOBAL REPRODUCIBILITY & PRECISION
# =============================================================================
RANDOM_SEED = 42
FLOAT_PRECISION = (
    np.float64
)  # Strictly enforce double precision to minimize metric floor noise

# =============================================================================
# DIRECTORY STRUCTURE
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
IMAGES_DIR = os.path.join(INPUT_DIR, "images")

# Working Directory for Idea 36 (MB-GHE)
WORKING_DIR = "./working/idea_36"
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
SUBMISSION_DIR = "./submission"
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure necessary writable directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# DATASET PATHS
# =============================================================================
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# =============================================================================
# FEATURE VIEW CONFIGURATIONS
# =============================================================================
# Defines the distinct feature representations (Views) used by the experts.
VIEW_CONFIGS = {
    "global": {
        "description": "Provided 192 features (Margin, Shape, Texture)",
        "type": "tabular",
        "use_raw_images": False,
    },
    "macro": {
        "description": "Explicit Morphological Descriptors (Hu Moments + Scalars)",
        "type": "image",
        "use_raw_images": True,
        "descriptors": [
            "hu_moments",  # 7 invariant moments
            "area",  # Scalar
            "perimeter",  # Scalar
            "aspect_ratio",  # Scalar
            "solidity",  # Scalar
            "extent",  # Scalar
            "eccentricity",  # Scalar
        ],
    },
}

# =============================================================================
# GAUSSIAN BASIS CONFIGURATIONS (PREPROCESSING)
# =============================================================================
# Defines the transformation spaces (Bases) to project data into Normality.
BASIS_CONFIGS = {
    "parametric": {"type": "power", "method": "yeo-johnson", "standardize": True},
    "quantile_coarse": {
        "type": "quantile",
        "n_quantiles": 20,  # Constrained to prevent overfitting
        "output_distribution": "normal",
        "random_state": RANDOM_SEED,
    },
    "quantile_fine": {
        "type": "quantile",
        "n_quantiles": 100,  # More flexible, captures local density
        "output_distribution": "normal",
        "random_state": RANDOM_SEED,
    },
}

# =============================================================================
# ESTIMATOR CONFIGURATIONS (EXPERTS)
# =============================================================================
# Defines the solvers to be trained on each View-Basis combination.
ESTIMATOR_CONFIGS = [
    # Generative Expert 1: LDA with Fixed Shrinkage (Low)
    {
        "name": "lda_shrink_0.001",
        "type": "lda",
        "params": {"solver": "lsqr", "shrinkage": 0.001},
    },
    # Generative Expert 2: LDA with Fixed Shrinkage (Medium)
    {
        "name": "lda_shrink_0.01",
        "type": "lda",
        "params": {"solver": "lsqr", "shrinkage": 0.01},
    },
    # Generative Expert 3: LDA with OAS Covariance Estimator
    {
        "name": "lda_oas",
        "type": "lda",
        "params": {
            "solver": "lsqr",
            "covariance_estimator": "oas",  # Special flag for implementation to use OAS()
        },
    },
    # Discriminative Expert: Logistic Regression with Dense Grid
    {
        "name": "logreg_cv",
        "type": "logreg_cv",
        "params": {
            "penalty": "l2",
            "solver": "lbfgs",
            "cv": 5,
            "scoring": "neg_log_loss",
            "max_iter": 5000,
            "n_jobs": -1,
            "Cs": 20,  # 20 values in log space
        },
    },
]

# =============================================================================
# ENSEMBLE SELECTION CONFIGURATION
# =============================================================================
SELECTION_CONFIG = {
    "method": "greedy_forward",
    "metric": "neg_log_loss",
    "max_experts": 50,  # Maximum number of experts to select
    "tolerance": 1e-7,  # Minimum improvement required to add an expert
    "replacement": True,  # Allow weighted selection (selecting same expert multiple times)
}


# =============================================================================
# RUNTIME CONFIGURATION
# =============================================================================
class RuntimeConfig:
    """
    Manages runtime parameters including debugging and parallelism.
    """

    def __init__(self, debug=False, n_jobs=12):
        self.debug = debug
        self.n_jobs = n_jobs

        # In debug mode, limit data size and iterations
        if self.debug:
            self.subsample_size = 100
            self.cv_folds = 2
        else:
            self.subsample_size = None  # Use full dataset
            self.cv_folds = 5

    def get_job_kwargs(self):
        return {"n_jobs": self.n_jobs}

    def __repr__(self):
        return f"RuntimeConfig(debug={self.debug}, n_jobs={self.n_jobs})"
