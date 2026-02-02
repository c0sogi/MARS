import os
import torch
import random
import numpy as np


class Config:
    """
    Central configuration for the Dual-Backbone Conditional Cascade Network (DB-CCN).
    Defines paths, model architecture, and training hyperparameters.
    """

    # ==========================================
    # 1. ENVIRONMENT & REPRODUCIBILITY
    # ==========================================
    SEED = 42
    NUM_WORKERS = 12
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # 2. FILE PATHS
    # ==========================================
    # Input Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working Directory (Cache)
    WORKING_DIR = "./working/idea_14"

    # Submission Directory
    SUBMISSION_DIR = "./submission"

    # Source Files
    TRAIN_BSON = os.path.join(INPUT_DIR, "train.bson")
    TEST_BSON = os.path.join(INPUT_DIR, "test.bson")
    CATEGORY_NAMES = os.path.join(INPUT_DIR, "category_names.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files (Pre-generated)
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Files (Parquet/NPY)
    # Stores the mapping from category_id to Level 1/2/3 labels
    HIERARCHY_MAPPING_PATH = os.path.join(WORKING_DIR, "hierarchy_mapping.parquet")

    # Feature Cache Paths (Decoupled Training)
    TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.npy")
    TRAIN_LABELS_PATH = os.path.join(
        WORKING_DIR, "train_labels.npy"
    )  # Stores target indices

    VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.npy")
    VAL_LABELS_PATH = os.path.join(WORKING_DIR, "val_labels.npy")

    TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.npy")
    TEST_IDS_PATH = os.path.join(WORKING_DIR, "test_ids.npy")

    # ==========================================
    # 3. DATA SPECIFICATIONS
    # ==========================================
    IMG_SIZE = 180

    # Hierarchy Counts (Derived from EDA)
    NUM_CLASSES_L1 = 49
    NUM_CLASSES_L2 = 483
    NUM_CLASSES_L3 = 5270

    # ==========================================
    # 4. MODEL ARCHITECTURE
    # ==========================================
    # Backbone Feature Dimensions
    RESNET_DIM = 2048
    EFFICIENTNET_DIM = 1280
    TOTAL_FEATURE_DIM = RESNET_DIM + EFFICIENTNET_DIM  # 3328

    # Cascade MLP Hidden Dimensions
    HIDDEN_DIM_L1 = 1024
    HIDDEN_DIM_L2 = 1024
    DROPOUT = 0.3

    # ==========================================
    # 5. TRAINING HYPERPARAMETERS
    # ==========================================
    # Feature Extraction
    BATCH_SIZE_EXTRACT = 256  # Batch size for image backbone inference

    # MLP Training
    BATCH_SIZE_TRAIN = 2048  # Large batch size for MLP training on cached features
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 20
    EARLY_STOPPING_PATIENCE = 5

    # Regularization
    LABEL_SMOOTHING = 0.1
    MIXUP_ALPHA = 0.2

    # Ensemble Strategy
    NUM_MODELS = 3

    # ==========================================
    # 6. DEBUGGING / DEVELOPMENT
    # ==========================================
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100000  # Subset size for rapid prototyping if DEBUG is True

    @staticmethod
    def setup():
        """
        Initializes the environment:
        1. Creates working and submission directories.
        2. Sets random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

        # Set seeds
        random.seed(Config.SEED)
        np.random.seed(Config.SEED)
        torch.manual_seed(Config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(Config.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
