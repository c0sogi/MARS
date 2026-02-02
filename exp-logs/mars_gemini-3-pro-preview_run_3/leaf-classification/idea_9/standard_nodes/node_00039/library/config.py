import os
import torch


class Config:
    """
    Configuration class for the Leaf Species Classification project.
    Implements settings for the Triple-Stream Modality-Specific LDA Ensemble.
    """

    # ==========================================
    # 1. File Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata CSVs (Pre-generated)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Working Directories
    WORKING_DIR = "./working"
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_9")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 2. General Settings
    # ==========================================
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debugging / Development
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 50  # Subset size when DEBUG is True

    # ==========================================
    # 3. Data Preprocessing
    # ==========================================
    IMG_SIZE = 224

    # ImageNet Normalization Constants
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # Multi-view Augmentation (Canonical Rotations)
    # 0, 90, 180, 270 degrees to enforce rotation invariance via averaging
    ROTATIONS = [0, 90, 180, 270]

    # ==========================================
    # 4. Model Architecture (Feature Extractors)
    # ==========================================
    # HuggingFace Hub Model IDs
    # Stream A: Global Geometry
    MODEL_DINO = "facebook/dinov2-large"

    # Stream B: Local Texture/Margin
    MODEL_CONVNEXT = "facebook/convnext-large-224-22k-1k"

    BATCH_SIZE = 32

    # ==========================================
    # 5. Dimensionality Reduction & Classification
    # ==========================================
    # PCA Settings (Visual Streams)
    PCA_VARIANCE = 0.99  # Retain 99% of variance

    # LDA Settings
    LDA_SOLVER = "lsqr"  # Least squares solution
    LDA_SHRINKAGE = "auto"  # Ledoit-Wolf shrinkage for HDLSS stability

    # Cross-Validation
    N_FOLDS = 5

    # Post-processing / Metrics
    PROB_CLIP_MIN = 1e-15
    PROB_CLIP_MAX = 1.0 - 1e-15

    @classmethod
    def setup(cls, seed=None):
        """
        Initialize the environment:
        1. Create necessary directories.
        2. Set random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set Seed
        effective_seed = seed if seed is not None else cls.SEED
        os.environ["PYTHONHASHSEED"] = str(effective_seed)
        torch.manual_seed(effective_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(effective_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
