import os
import torch


class Config:
    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Utilizing all available vCPUs
    NUM_WORKERS = 12

    # ==========================================
    # Directory & File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Working directory for this specific experimental run
    WORK_DIR = "./working/idea_5"
    SUBMISSION_DIR = "./submission"

    # Metadata CSVs (Pre-generated)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Raw Metadata (for extracting taxonomy hierarchy)
    TRAIN_METADATA_JSON = os.path.join(INPUT_DIR, "train/metadata.json")

    # Output & Cache Paths
    # Stores the mapping between Species -> Genus/Family for hierarchical learning
    TAXONOMY_MAP_PATH = os.path.join(WORK_DIR, "taxonomy_mappings.parquet")
    # Checkpoints
    CHECKPOINT_PATH = os.path.join(WORK_DIR, "checkpoint.pth")
    BEST_MODEL_PATH = os.path.join(WORK_DIR, "best_model.pth")
    # Final Submission
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Parameters
    # ==========================================
    IMAGE_SIZE = 224
    NUM_CLASSES = 64500  # Total species count from data analysis

    # Normalization Statistics
    # Derived from Data Analysis (reflects white background of herbarium sheets)
    MEAN = [0.7785, 0.7599, 0.7235]
    STD = [0.2862, 0.2921, 0.2973]

    # Debugging: Set to an integer (e.g., 5000) to train on a subset, or None for full data
    DEBUG_SAMPLE_SIZE = None

    # ==========================================
    # Model Parameters
    # ==========================================
    BACKBONE = "convnext_tiny"
    PRETRAINED = True
    DROPOUT = 0.0
    DROP_PATH_RATE = 0.1

    # ==========================================
    # Training Parameters
    # ==========================================
    # Batch size optimized for A100 40GB with ConvNeXt-Tiny
    BATCH_SIZE = 256

    # Stage 1: Representation Learning
    # Objective: Learn features using Instance-Balanced sampling
    STAGE1_EPOCHS = 8
    STAGE1_LR = 1e-3  # Max LR for OneCycle
    LABEL_SMOOTHING = 0.1

    # Stage 2: Classifier Re-balancing
    # Objective: Fine-tune heads with Class-Balanced sampling (Backbone frozen)
    STAGE2_EPOCHS = 2
    STAGE2_LR = 1e-4

    # Optimizer Settings
    WEIGHT_DECAY = 1e-2

    # ==========================================
    # Utility Methods
    # ==========================================
    @classmethod
    def setup(cls):
        """Creates necessary working directories."""
        os.makedirs(cls.WORK_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def override(cls, **kwargs):
        """
        Updates configuration parameters dynamically.
        Useful for changing epochs or debug sizes programmatically.
        """
        for k, v in kwargs.items():
            if hasattr(cls, k):
                setattr(cls, k, v)
