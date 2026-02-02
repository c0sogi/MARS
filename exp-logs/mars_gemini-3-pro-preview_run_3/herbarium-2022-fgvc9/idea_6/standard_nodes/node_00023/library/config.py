import os
import torch


class Config:
    """
    Global configuration for the Hierarchical Multi-Task EfficientNetV2-Small pipeline.
    Implements the strategy for Idea 6: Corrected Hierarchical EfficientNetV2 with Progressive Resolution.
    """

    # -------------------------------------------------------------------------
    # General Settings
    # -------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 5000

    # Compute
    NUM_WORKERS = 12  # Utilizing available vCPUs
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific idea (Idea 6)
    WORKING_DIR = "./working/idea_6"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Submission directory
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    # Pre-generated metadata splits
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Raw JSON for extracting taxonomic hierarchy (Family -> Genus -> Species)
    HIERARCHY_JSON_PATH = os.path.join(INPUT_DIR, "train_metadata.json")

    # Cache file for processed hierarchy mappings (Parquet format)
    HIERARCHY_MAPPING_PATH = os.path.join(WORKING_DIR, "hierarchy_mappings_v2.parquet")

    # Model Checkpoints
    CHECKPOINT_STAGE_1 = os.path.join(WORKING_DIR, "stage_1_best.pth")
    CHECKPOINT_STAGE_2 = os.path.join(WORKING_DIR, "stage_2_best.pth")

    # Final Submission
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    MODEL_NAME = "tf_efficientnetv2_s.in1k"  # timm backbone
    PRETRAINED = True

    # Target Classes
    NUM_CLASSES_SPECIES = 15501
    # Note: Family and Genus class counts are derived dynamically from the hierarchy mapping

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    # Optimizer & Scheduler
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Loss Function Configuration
    LABEL_SMOOTHING = 0.1
    # Multi-Task Loss Weights
    WEIGHT_SPECIES = 1.0
    WEIGHT_GENUS = 0.1
    WEIGHT_FAMILY = 0.1

    # Progressive Resolution Strategy

    # Stage 1: Feature Learning (Lower Res, Higher Batch Size)
    STAGE_1_RES = 224
    STAGE_1_BATCH_SIZE = 32  # Fits on T4 16GB
    STAGE_1_EPOCHS = 12

    # Stage 2: Fine-Grained Refinement (Higher Res, Lower Batch Size)
    STAGE_2_RES = 320
    STAGE_2_BATCH_SIZE = 16
    STAGE_2_EPOCHS = 8

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------
    TTA_FLIP = True  # Enable Horizontal Flip Test Time Augmentation

    @staticmethod
    def set_seed(seed=42):
        """
        Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
        """
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
