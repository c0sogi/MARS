import os
import torch
import random
import numpy as np


class Config:
    # ==========================================
    # 1. File Paths & Directories
    # ==========================================
    # Root directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Specific Metadata Files (Generated previously)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # DICOM Directories
    DICOM_TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    DICOM_TEST_DIR = os.path.join(INPUT_DIR, "test")

    # Output Directories
    # Specific cache directory for this idea iteration
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_74")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 2. Data Processing Parameters
    # ==========================================
    # Image Generation: Fixed Overlapping Orthogonal Tri-Slabs
    IMG_SIZE = 224  # Native resolution for EfficientNet-B0
    SLAB_COUNT = 3  # Number of slabs per view
    SLAB_OVERLAP = 0.15  # 15% overlap between slabs

    # Normalization constants (ImageNet defaults)
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]

    # DataLoader settings
    BATCH_SIZE = 32  # Adjusted for A100 memory with dual backbones
    NUM_WORKERS = 4  # Based on 12 vCPUs

    # Debugging/Development
    DEBUG = False  # Set to True to use a small subset of data
    DEBUG_SAMPLE_SIZE = 20  # Number of patients to use in debug mode

    # ==========================================
    # 3. Model Architecture Parameters
    # ==========================================
    # Visual Backbone
    BACKBONE_NAME = "efficientnet_b0"
    BACKBONE_DIM = 1280  # Output dimension of EfficientNet-B0 GAP
    PRETRAINED = True

    # Tabular/Latent
    TABULAR_LATENT_DIM = 128  # Dimension for Shared Latent Vector (T_lat)
    CONTEXT_DIM = 128  # Dimension for Context Vector (H_ctx)

    # Architecture specifics
    DROPOUT_RATE = 0.1
    ATTENTION_HEADS = 4  # For the Contextualization Phase
    FFN_DIM = 512  # Feed-forward dimension in attention block

    # ==========================================
    # 4. Training Hyperparameters
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS
    ETA_MIN = 1e-6

    # Early Stopping
    PATIENCE = 8  # Strict patience as per design

    # Metric Constraints
    MAX_ERROR = 1000  # Clipping threshold for metric calculation
    MIN_CONFIDENCE = 70  # Clipping threshold for confidence

    @classmethod
    def setup(cls):
        """
        Creates necessary output directories and sets random seeds.
        """
        # Create directories
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set reproducibility
        cls.seed_everything(cls.SEED)

        print(f"Configuration setup complete.")
        print(f"Device: {cls.DEVICE}")
        print(f"Cache Directory: {cls.CACHE_DIR}")

    @staticmethod
    def seed_everything(seed):
        """
        Seeds all random number generators for reproducibility.
        """
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
