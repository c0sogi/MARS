import os
import torch
import random
import numpy as np


class Config:
    """
    Central configuration for the HubMap FTU Detection task.
    Implements parameters for U-Net++ with ConvNeXt-Tiny backbone,
    Progressive Resizing strategy, and Fail-Open ROI logic.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    PROJECT_NAME = "HubMap_FTU_Detection"
    IDEA_NAME = "idea_7"
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 16  # Number of images to use in debug mode

    # =========================================================================
    # Directories & Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific idea/experiment
    WORKING_DIR = os.path.join("./working", IDEA_NAME)
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join("./submission", "submission.csv")

    # Metadata Files (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # =========================================================================
    # Model Architecture
    # =========================================================================
    # Using segmentation_models_pytorch naming convention
    ARCH = "UnetPlusPlus"
    ENCODER_NAME = "tu-convnext_tiny"  # ConvNeXt-Tiny backbone
    ENCODER_WEIGHTS = "imagenet"
    IN_CHANNELS = 3  # Raw RGB input
    CLASSES = 1  # Binary segmentation (FTU vs Background)
    ACTIVATION = None  # Output logits for BCEWithLogitsLoss

    # Decoder settings
    DECODER_CHANNELS = [256, 128, 64, 32, 16]

    # Deep Supervision
    DEEP_SUPERVISION = True
    # Loss weights for multi-scale outputs (deepest to shallowest)
    LOSS_WEIGHTS = [1.0, 0.5, 0.25, 0.125]

    # =========================================================================
    # Training Strategy: Progressive Resizing
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 12  # Utilize available vCPUs

    # Optimization
    WEIGHT_DECAY = 1e-5
    SCHEDULER_T0 = 10
    SCHEDULER_T_MULT = 2
    MIN_LR = 1e-6
    EARLY_STOPPING_PATIENCE = 6

    # Phase 1: High Throughput (Coarse learning)
    PHASE1 = {
        "EPOCHS": 20,
        "BATCH_SIZE": 32,  # Larger batch size for stability
        "TILE_SIZE": 512,  # Smaller resolution for speed
        "LR": 3e-4,  # Standard learning rate
    }

    # Phase 2: High Precision (Fine-tuning)
    PHASE2 = {
        "EPOCHS": 20,
        "BATCH_SIZE": 8,  # Smaller batch size due to memory constraints
        "TILE_SIZE": 768,  # High resolution for detail
        "LR": 5e-5,  # Lower learning rate for fine-tuning
    }

    # =========================================================================
    # Data Processing & Augmentation
    # =========================================================================
    # Tiling parameters
    TRAIN_OVERLAP = 0.5
    TISSUE_AREA_THRESHOLD = 0.05  # Only keep tiles with >5% tissue

    # Augmentation
    AUG_PROB = 0.5  # Probability for geometric augmentations

    # =========================================================================
    # Inference & Post-Processing
    # =========================================================================
    # Fail-Open Logic: If anatomical mask is missing, use full image
    FAIL_OPEN_ROI = True

    # Tiling during inference
    INFERENCE_OVERLAP = 0.5  # Significant overlap for Gaussian stitching

    # Thresholds
    PREDICTION_THRESHOLD = 0.5  # Binary threshold for logits (after sigmoid)
    MIN_PIXEL_SIZE = 50  # Remove connected components smaller than this

    @classmethod
    def set_seed(cls, seed=42):
        """Sets the random seed for reproducibility."""
        cls.SEED = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # Ensure deterministic behavior
