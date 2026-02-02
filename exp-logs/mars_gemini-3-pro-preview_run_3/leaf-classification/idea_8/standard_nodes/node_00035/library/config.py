import os
import torch


class Config:
    """
    Configuration class for the Random Subspace Ensemble of LDA
    with Multi-View Self-Supervised Features pipeline.
    """

    # ==========================================
    # Paths
    # ==========================================
    INPUT_DIR = "./input"
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")

    METADATA_DIR = "./metadata"
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    WORKING_DIR = "./working"
    # Specific cache directory as required for deterministic data processing
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_9")

    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data & Preprocessing
    # ==========================================
    SEED = 42
    IMAGE_SIZE = 224
    NUM_CLASSES = 99

    # Multi-view settings: 0, 90, 180, 270 degrees
    ROTATION_ANGLES = [0, 90, 180, 270]

    # ==========================================
    # Feature Extraction Models
    # ==========================================
    # Global Geometry Stream (ViT-Large)
    DINO_MODEL_NAME = "facebook/dinov2-large"

    # Local Texture Stream (ConvNeXt Large)
    # Using timm naming convention
    CONVNEXT_MODEL_NAME = "convnext_large.fb_in22k_ft_in1k"

    # ==========================================
    # Dimensionality Reduction & Ensemble
    # ==========================================
    # PCA Variance retention threshold
    PCA_VARIANCE = 0.99

    # LDA Solver settings
    LDA_SOLVER = "lsqr"
    LDA_SHRINKAGE = "auto"  # Ledoit-Wolf shrinkage

    # ==========================================
    # Hardware & Execution
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 32
    NUM_WORKERS = 4

    @classmethod
    def setup(cls):
        """
        Ensures necessary directories exist.
        Should be called at the start of the pipeline.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set deterministic behavior for reproducibility
        os.environ["PYTHONHASHSEED"] = str(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
