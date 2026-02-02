import os
import torch
import numpy as np
import random


class Config:
    # ==========================
    # General Settings
    # ==========================
    PROJECT_NAME = "Cervical_Spine_Fracture_Detection"
    IDEA_NAME = "idea_13"
    SEED = 42
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 20  # Small subset for debugging pipeline
    NUM_WORKERS = 4  # Optimized for 12 vCPUs

    # ==========================
    # Data Paths
    # ==========================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata files (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Raw Data
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")
    BOUNDING_BOX_PATH = os.path.join(INPUT_DIR, "train_bounding_boxes.csv")
    SEGMENTATION_DIR = os.path.join(INPUT_DIR, "segmentations")

    # Working Directory for Outputs and Cache
    # We use idea_13 specific folder
    WORKING_DIR = os.path.join("./working", IDEA_NAME)
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    MODEL_DIR = os.path.join(WORKING_DIR, "models")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure writeable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================
    # Model Architecture
    # ==========================
    # Backbone: EfficientNet-B4 as requested
    BACKBONE = "tf_efficientnet_b4_ns"

    # Input Dimensions
    IN_CHANNELS = 3  # 2.5D Stacking (z-1, z, z+1)
    IMAGE_SIZE = (384, 384)  # Resolution appropriate for B4

    # Sequence Modeling
    SEQ_LEN = 96  # High-density sampling to capture small fractures
    LSTM_HIDDEN_SIZE = 512
    LSTM_LAYERS = 2
    BIDIRECTIONAL = True

    # Heads
    NUM_CLASSES = 8  # C1, C2, C3, C4, C5, C6, C7, patient_overall

    # ==========================
    # Training Hyperparameters
    # ==========================
    EPOCHS = 10
    # Batch size is constrained by SEQ_LEN=96 * EfficientNet-B4 * 384x384
    BATCH_SIZE = 2
    GRAD_ACCUM_STEPS = 8  # Effective batch size = 16

    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    MAX_GRAD_NORM = 10.0

    # Loss Configuration
    # No positive class weighting to ensure calibrated probabilities
    POS_WEIGHT = 1.0

    # Supervised Attention Guidance
    ATTENTION_LAMBDA = 1.0  # Weight for the auxiliary attention loss

    # Early Stopping
    PATIENCE = 3
    MIN_DELTA = 1e-4

    # ==========================
    # Normalization
    # ==========================
    # Standard ImageNet statistics
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    @staticmethod
    def setup_reproducibility(seed=42):
        """
        Sets the seed for reproducibility across random, numpy, and torch.
        """
        Config.SEED = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        os.environ["PYTHONHASHSEED"] = str(seed)
        print(f"Reproducibility seeds set to {seed}")
