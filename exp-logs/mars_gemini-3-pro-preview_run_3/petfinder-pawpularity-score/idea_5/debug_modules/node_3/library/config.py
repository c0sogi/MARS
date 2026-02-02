import os
import torch


class Config:
    # ==========================================
    # 1. Paths and Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching intermediate files (features, models)
    # Using 'idea_5' as the specific experiment identifier
    WORKING_DIR = "./working/idea_5"
    SUBMISSION_DIR = "./submission"

    # Metadata Files
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_meta.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_meta.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_meta.csv")

    # Output Submission File
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 2. Global Hyperparameters
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for testing

    # Image Processing
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    NUM_WORKERS = 4

    # Computation
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # 3. Model Architecture Specifications
    # ==========================================
    # List of backbones to be used for feature extraction.
    # We use timm library model names.
    BACKBONES = [
        # Swin Transformer (Supervised) - Captures global composition
        "swin_base_patch4_window7_224",
        # EfficientNetV2 (Supervised) - Captures local texture/details
        "tf_efficientnetv2_s",
        # DINOv2 (Self-Supervised) - Captures geometry/correspondence
        "vit_base_patch14_dinov2.lvd142m",
    ]

    # ==========================================
    # 4. Feature Processing & Stacking
    # ==========================================
    # PCA Variance retention threshold for independent component compression
    PCA_VARIANCE = 0.95

    # Cross-Validation
    N_FOLDS = 5

    # Stacking Level 1 Models
    # SVR Kernel
    SVR_C = 1.0
    SVR_EPSILON = 0.1

    # LightGBM Params (General)
    LGBM_PARAMS = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.01,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "n_estimators": 1000,
        "verbose": -1,
        "random_state": SEED,
        "n_jobs": -1,
    }

    # ==========================================
    # 5. Setup Logic
    # ==========================================
    @classmethod
    def setup(cls):
        """Creates necessary output directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Run setup immediately when module is imported to ensure directories exist
Config.setup()
