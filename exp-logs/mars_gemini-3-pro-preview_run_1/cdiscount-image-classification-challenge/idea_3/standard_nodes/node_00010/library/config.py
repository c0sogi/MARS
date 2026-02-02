import os
import torch


class Config:
    """
    Configuration for the Product Categorization Pipeline.
    """

    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42

    # ==========================================
    # Directory & File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_3"
    SUBMISSION_DIR = "./submission"

    # Raw BSON Data
    TRAIN_BSON = os.path.join(INPUT_DIR, "train.bson")
    TEST_BSON = os.path.join(INPUT_DIR, "test.bson")

    # Metadata CSVs (Pre-generated)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Auxiliary Data
    CATEGORY_NAMES = os.path.join(INPUT_DIR, "category_names.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Outputs
    # Path to save the best model checkpoint
    CHECKPOINT_PATH = os.path.join(WORKING_DIR, "resnet50_mil_best.pth")
    # Path to save the processed category hierarchy cache
    HIERARCHY_CACHE_PATH = os.path.join(WORKING_DIR, "category_hierarchy.parquet")
    # Path for the final submission file
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Hyperparameters
    # ==========================================
    IMG_SIZE = 180
    NUM_WORKERS = 12
    MAX_IMGS_PER_PRODUCT = 4  # Dataset contains 1-4 images per product

    # Debugging flags to control dataset size
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 5000

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    BACKBONE = "resnet50"

    # Hierarchical Class Counts (derived from analysis)
    NUM_CLASSES_L1 = 49
    NUM_CLASSES_L2 = 483
    NUM_CLASSES_L3 = 5270  # Target fine-grained categories

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Batch size (Number of products).
    # 256 products * ~2 avg images = ~512 images per batch.
    # Fits comfortably in A100 40GB VRAM with ResNet50.
    BATCH_SIZE = 256

    NUM_EPOCHS = 3
    LEARNING_RATE = 0.01  # Peak LR for OneCycle Scheduler
    WEIGHT_DECAY = 1e-4
    LABEL_SMOOTHING = 0.1

    # Loss Weights for Hierarchical Supervision
    LOSS_WEIGHT_L3 = 1.0  # Fine-grained (Target)
    LOSS_WEIGHT_L2 = 0.3  # Mid-level
    LOSS_WEIGHT_L1 = 0.1  # Coarse-level

    @classmethod
    def setup(cls):
        """
        Ensures that the working and submission directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories on module import
Config.setup()
