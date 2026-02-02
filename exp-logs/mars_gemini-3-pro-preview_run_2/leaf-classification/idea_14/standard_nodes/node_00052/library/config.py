import os
import numpy as np


class Config:
    """
    Global configuration for the Deep-Feature Augmented Multi-View Ensemble (Idea 14).
    """

    # =========================================================================
    # DIRECTORIES AND PATHS
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")

    # Working directory for caching intermediate results (e.g., extracted deep features)
    WORKING_DIR = "./working/idea_14"

    # Directory for final submission files
    SUBMISSION_DIR = "./submission"

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # GLOBAL SETTINGS
    # =========================================================================
    RANDOM_SEED = 42

    # =========================================================================
    # IMAGE PROCESSING (VIEW 2)
    # =========================================================================
    # Dimensions for ResNet18 input
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

    # Batch size for feature extraction (inference only)
    BATCH_SIZE = 32

    # =========================================================================
    # FEATURE ENGINEERING
    # =========================================================================
    # Variance ratio to retain when applying PCA to deep embeddings
    PCA_VARIANCE = 0.95

    # =========================================================================
    # MODEL HYPERPARAMETERS
    # =========================================================================
    # Cross-Validation settings
    CV_FOLDS = 3

    # Logistic Regression (Estimators A & C)
    # Dense, broad logarithmic grid for C (Inverse of regularization strength)
    # Synthesizes the need for broad search with high density for sharp optima.
    LR_CS = np.logspace(-4, 4, 50)

    LR_SOLVER = "lbfgs"
    LR_MAX_ITER = 10000  # High iteration count to ensure convergence
    LR_PENALTY = "l2"
    LR_SCORING = "neg_log_loss"  # Explicitly optimize for log loss

    # Linear Discriminant Analysis (Estimator B)
    LDA_SOLVER = "lsqr"  # Required for shrinkage
    LDA_SHRINKAGE = "auto"  # Ledoit-Wolf shrinkage for stability

    # =========================================================================
    # SUBMISSION FORMATTING
    # =========================================================================
    # Clipping bounds to avoid log(0) extremes in the metric
    PROB_CLIP_MIN = 1e-15
    PROB_CLIP_MAX = 1.0 - 1e-15
