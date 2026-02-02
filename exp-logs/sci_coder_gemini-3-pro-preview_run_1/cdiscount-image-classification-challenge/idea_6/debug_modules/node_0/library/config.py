import os
import torch


class Config:
    """
    Configuration class for the Multi-Level ResNet-50 Image Categorization Task.
    Stores file paths, hyperparameters, and model settings.
    """

    # ==========================================
    # Directories and Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_6"
    SUBMISSION_DIR = "./submission"

    # Input Files
    TRAIN_BSON = os.path.join(INPUT_DIR, "train.bson")
    TEST_BSON = os.path.join(INPUT_DIR, "test.bson")
    CATEGORY_NAMES = os.path.join(INPUT_DIR, "category_names.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files (Pre-generated)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_CHECKPOINT = os.path.join(WORKING_DIR, "resnet50_multi_level_best.pth")

    # Cache Files
    # Stores the mapping from category_id to level1, level2 indices
    HIERARCHY_MAPPING_PATH = os.path.join(WORKING_DIR, "hierarchy_mappings.parquet")

    # ==========================================
    # Data Specifications
    # ==========================================
    IMG_SIZE = 180
    CHANNELS = 3

    # Class Counts (derived from analysis)
    # Level 1: Broad categories (e.g., SPORT)
    NUM_CLASSES_L1 = 49
    # Level 2: Intermediate (e.g., CYCLES)
    NUM_CLASSES_L2 = 483
    # Level 3: Target fine-grained categories (e.g., VELO ENFANT)
    NUM_CLASSES_L3 = 5270

    # BSON Parsing Constants
    BSON_TYPE_BINARY = 5

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 512  # High batch size to saturate A100
    NUM_EPOCHS = 2  # Sufficient for convergence on 7M dataset with OneCycleLR

    # Optimization
    BASE_LR = 1e-2  # Peak LR for OneCycleLR (will be scaled or tuned)
    WEIGHT_DECAY = 1e-4
    LABEL_SMOOTHING = 0.1

    # Loss Weights for Deep Supervision
    LOSS_WEIGHT_FINE = 1.0
    LOSS_WEIGHT_MID = 0.3  # Stage 4 features -> Level 2
    LOSS_WEIGHT_COARSE = 0.1  # Stage 3 features -> Level 1

    # ==========================================
    # Hardware & Reproducibility
    # ==========================================
    NUM_WORKERS = 12
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Creates necessary output directories.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set deterministic flags for PyTorch
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = (
                True  # Enable benchmark for speed on fixed size inputs
            )
