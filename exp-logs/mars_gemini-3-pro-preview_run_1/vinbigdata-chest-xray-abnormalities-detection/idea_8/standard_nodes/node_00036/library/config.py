import os
import torch


class Config:
    """
    Configuration class for the Thoracic Lung Disease Detection Task.
    Implements settings for the Task-Aligned Spatially-Decoupled Architecture.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False
    NUM_WORKERS = 12  # Utilizing all available vCPUs
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    # Input Directories (Read-Only)
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Files (Pre-generated)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_meta.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_meta.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_meta.csv")

    # Working Directories (Write Access)
    # Using 'idea_8' as the current iteration identifier
    WORK_DIR = "./working/idea_8"
    CACHE_DIR = os.path.join(WORK_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORK_DIR, "submission")

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    # Architecture: EfficientNet-B0 with BiFPN
    BACKBONE = "tf_efficientnet_b0_ns"  # timm implementation
    PRETRAINED = True

    # Input Resolution
    # 640x640 is chosen to balance detail for small nodules vs receptive field
    IMAGE_SIZE = 640

    # Classes
    # 14 Critical Findings (IDs 0-13)
    # Class 14 is reserved for "No finding"
    NUM_CLASSES = 14
    NO_FINDING_CLASS_ID = 14

    CLASS_ID_TO_NAME = {
        0: "Aortic enlargement",
        1: "Atelectasis",
        2: "Calcification",
        3: "Cardiomegaly",
        4: "Consolidation",
        5: "ILD",
        6: "Infiltration",
        7: "Lung Opacity",
        8: "Nodule/Mass",
        9: "Other lesion",
        10: "Pleural effusion",
        11: "Pleural thickening",
        12: "Pneumothorax",
        13: "Pulmonary fibrosis",
    }

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    # Training Duration
    # 20 Epochs required for convergence with strong augmentations
    EPOCHS = 20

    # Batch Size
    # EfficientNet-B0 is lightweight; 32 fits comfortably on A100
    # Reduced to 8 to fit 16GB GPU (Cite debug_lesson_7)
    BATCH_SIZE = 8

    # Optimization
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS
    MIN_LR = 1e-6

    # =========================================================================
    # Augmentation Settings
    # =========================================================================
    # Geometric
    MIN_VISIBILITY = 0.3  # Ensure at least 30% of bbox is visible after crop

    # =========================================================================
    # Inference & Post-Processing
    # =========================================================================
    # Global Gating
    # If Global Head predicts "No Finding" prob > 0.8, suppress all boxes
    GLOBAL_THRESHOLD = 0.8

    # Detection
    CONF_THRESHOLD = 0.001  # Low threshold to maximize recall for mAP calculation
    IOU_THRESHOLD = 0.4  # NMS IoU Threshold

    @classmethod
    def setup(cls):
        """
        Ensures all necessary working directories exist.
        Should be called at the start of any script using this config.
        """
        os.makedirs(cls.WORK_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Automatically setup directories when config is imported
Config.setup()
