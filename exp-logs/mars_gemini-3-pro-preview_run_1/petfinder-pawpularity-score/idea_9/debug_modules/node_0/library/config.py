import os
import torch
import numpy as np


class Config:
    """
    Configuration class for the Multi-Scale Tri-Paradigm Stacking Ensemble.
    """

    # ==========================================
    # General Setup
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True for fast debugging on a subset
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use when DEBUG is True

    # Compute configuration
    # 12 vCPUs available, 2-4 workers is typically optimal for DataLoader overhead
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Directories & Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_9"
    SUBMISSION_DIR = "./submission"

    # Ensure output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata File Paths
    # Using the pre-generated metadata files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Submission Output Path
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Processing & Image Strategy
    # ==========================================
    IMAGE_SIZE = 224

    # Dual-View Strategy:
    # 1. Global View: Resize full image to IMAGE_SIZE
    # 2. Local View: Central crop covering CROP_SCALE area, then resize to IMAGE_SIZE
    CROP_SCALE = 0.80

    # Feature Extraction
    BATCH_SIZE = 32

    # ==========================================
    # Model Backbones (HuggingFace Transformers)
    # ==========================================
    # Selecting orthogonal experts: Semantic (CLIP), Geometric (DINOv2), Textural (ConvNeXt)
    BACKBONES = {
        "clip": "openai/clip-vit-large-patch14",
        "dinov2": "facebook/dinov2-large",
        "convnext": "facebook/convnext-large-224-22k-1k",
    }

    # ==========================================
    # Level-0 Experts Hyperparameters
    # ==========================================

    # 1. Ridge Regression Expert
    # Using a wide range of alphas to handle varying degrees of multicollinearity
    # Logspace from 0.01 to ~50,000
    RIDGE_ALPHAS = np.logspace(-2, 4.7, 20).tolist()

    # 2. SVR Expert (RBF Kernel)
    # Requires StandardScaler on inputs
    SVR_C = 1.0
    SVR_EPSILON = 0.1
    SVR_KERNEL = "rbf"

    # 3. ExtraTrees Regressor Expert
    # Requires PCA on embeddings to manage dimensionality
    ET_N_ESTIMATORS = 200
    ET_MAX_DEPTH = 12
    ET_MIN_SAMPLES_SPLIT = 5
    ET_RANDOM_STATE = SEED
    ET_N_JOBS = -1

    # Feature Engineering for Tree Models
    # Reduce embedding dimension before concatenation with metadata
    PCA_COMPONENTS = 64

    # ==========================================
    # Level-1 Meta-Learner
    # ==========================================
    # Bayesian Ridge Regressor
    # Automatically infers regularization parameters
    META_N_ITER = 300
    META_TOL = 1e-3

    # ==========================================
    # Validation Strategy
    # ==========================================
    # 5-Fold Cross-Validation on the full dataset (Train + Val merged)
    N_FOLDS = 5

    # ==========================================
    # Utility Methods
    # ==========================================
    @classmethod
    def get_cache_path(cls, filename):
        """Generates a full path for a cache file in the working directory."""
        return os.path.join(cls.WORKING_DIR, filename)
