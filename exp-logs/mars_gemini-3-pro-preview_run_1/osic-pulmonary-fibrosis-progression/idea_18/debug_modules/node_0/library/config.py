import os
import torch
import random
import numpy as np


class Config:
    """
    Central configuration for the Lung Function Decline Prediction task.
    Implements settings for the Calibrated Symmetric Dual-Axis Network.
    """

    # ==========================================
    # 1. General Settings & Paths
    # ==========================================
    PROJECT_NAME = "lung_function_decline"
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 20

    # Directory Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Specific Cache Directory for this Idea (Idea 18)
    # Used for storing processed Tri-Slab images (npy/parquet)
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_18")

    # Metadata File Paths
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Submission Output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 2. Data Preprocessing & Augmentation
    # ==========================================
    # Image Dimensions
    IMG_SIZE = 224  # Native resolution to avoid upscaling artifacts

    # Tri-Slab Generation
    # Overlap percentage for the fixed boundaries (0-33%, 33-66%, 66-100%)
    SLAB_OVERLAP = 0.15

    # Normalization Constants (approximate for CT Hounsfield Units mapped to 0-255)
    # Using ImageNet stats as we initialize with ImageNet weights
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # ==========================================
    # 3. Model Architecture
    # ==========================================
    # Backbone
    BACKBONE_NAME = "efficientnet_b0"
    BACKBONE_DIM = 1280  # Native dimensionality of EfficientNet-B0 GAP output
    PRETRAINED = True

    # Tabular Embedding
    TABULAR_INPUT_DIM = 7  # Age, Sex(enc), Smoking(enc), Percent, etc.
    TABULAR_HIDDEN_DIM = 512

    # Fusion & Calibration
    # The dimension to which tabular data is up-projected to match visual features
    FUSION_DIM = 1280
    NUM_ATTENTION_HEADS = 4
    DROPOUT_RATE = 0.2

    # ==========================================
    # 4. Training Hyperparameters
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Batch Size & Workers
    BATCH_SIZE = 32
    NUM_WORKERS = 4

    # Optimization
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS
    ETA_MIN = 1e-6

    # Early Stopping
    # Strict patience to prevent overfitting on small dataset
    PATIENCE = 8

    # Metric / Loss
    # Threshold for clipping absolute error in metric calculation (ml)
    ERR_CLIP_THRESHOLD = 1000
    # Confidence clipping (ml)
    CONFIDENCE_CLIP = 70

    @classmethod
    def setup_directories(cls):
        """Creates necessary working directories."""
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


def setup_reproducibility(seed=42):
    """
    Sets random seeds for Python, NumPy, and PyTorch to ensure
    reproducible results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
