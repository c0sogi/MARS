import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    """
    Central configuration for the Tri-Stream Context-Gated Network (TSCG-Net) pipeline.
    """

    # ==========================
    # General Settings
    # ==========================
    SEED = 42
    DEBUG = False  # Set True to use a small subset of data for debugging
    EXPERIMENT_NAME = "idea_67_TSCG_Net"

    # ==========================
    # Paths
    # ==========================
    # Input Directories (Read-Only)
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Working Directories (Write Access)
    WORKING_DIR = "./working"
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_67")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================
    # Data Preprocessing
    # ==========================
    IMG_SIZE = 224  # Native resolution for EfficientNet-B0
    NUM_SLABS = 3  # Tri-Slab generation
    SLAB_OVERLAP = 0.15  # 15% overlap between slabs

    # ImageNet Normalization Constants
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # ==========================
    # Model Architecture
    # ==========================
    BACKBONE = "efficientnet_b0"
    PRETRAINED = True

    # Dimensionality settings for the Tri-Stream architecture
    VISUAL_BACKBONE_DIM = 1280  # Output of EfficientNet-B0 GAP
    LATENT_DIM = 128  # Shared Latent Tabular Encoder output (Stream 3)

    # Fusion Stream Dimensions
    VISUAL_CONTEXT_DIM = 128  # Stream 1: Projected Visual Context
    TABULAR_CONTEXT_DIM = 64  # Stream 2: Projected Tabular Context

    # Final assembled vector dimension: 128 (Vis) + 64 (Ctx) + 128 (Prior)
    FUSION_DIM = VISUAL_CONTEXT_DIM + TABULAR_CONTEXT_DIM + LATENT_DIM

    # ==========================
    # Training Hyperparameters
    # ==========================
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # Scheduler (Cosine Annealing)
    T_MAX = 50
    ETA_MIN = 1e-6

    # Early Stopping
    PATIENCE = 8

    # Hardware
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================
    # Metric / Inference
    # ==========================
    MAX_ERROR = 1000  # Threshold for metric calculation
    CONFIDENCE_CLIP = 70  # Minimum confidence for metric calculation

    @classmethod
    def setup(cls):
        """
        Initializes the environment by creating necessary directories.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def get_parameter_summary(cls):
        """Returns a dictionary of key parameters for logging."""
        return {
            "Experiment": cls.EXPERIMENT_NAME,
            "Image Size": cls.IMG_SIZE,
            "Batch Size": cls.BATCH_SIZE,
            "LR": cls.LEARNING_RATE,
            "Device": cls.DEVICE,
            "Debug": cls.DEBUG,
        }


# Initialize environment on import
seed_everything(Config.SEED)
Config.setup()
