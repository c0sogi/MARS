import os
import torch
import random
import numpy as np


class Config:
    # ==========================================
    # PATHS
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_8"

    # Raw Data
    TRAIN_BSON = os.path.join(INPUT_DIR, "train.bson")
    TEST_BSON = os.path.join(INPUT_DIR, "test.bson")
    CATEGORY_NAMES = os.path.join(INPUT_DIR, "category_names.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths / Cache
    HIERARCHY_MAP_PATH = os.path.join(WORKING_DIR, "hierarchy_map.parquet")

    # Cached Features (Decoupled Training)
    TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.npy")
    TRAIN_LABELS_PATH = os.path.join(
        WORKING_DIR, "train_labels.npy"
    )  # Stores hierarchical labels

    VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.npy")
    VAL_LABELS_PATH = os.path.join(WORKING_DIR, "val_labels.npy")

    TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.npy")
    TEST_IDS_PATH = os.path.join(WORKING_DIR, "test_ids.npy")

    # Model & Submission
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "hierarchical_resnet50_mlp.pth")
    SUBMISSION_PATH = "submission.csv"

    # ==========================================
    # SYSTEM
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # 12 vCPUs available
    NUM_WORKERS = 12

    # ==========================================
    # DATASET HYPERPARAMETERS
    # ==========================================
    # Set DEBUG to True to run on a small subset for testing pipeline logic
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 50000

    # Image Processing
    IMG_SIZE = 180  # Native size in BSON
    RESIZE_SIZE = 224  # ResNet standard input

    # ==========================================
    # MODEL HYPERPARAMETERS
    # ==========================================
    BACKBONE = "resnet50"
    FEATURE_DIM = 2048

    # Hierarchy dimensions (based on category_names.csv)
    # Level 1: Coarse Category
    NUM_CLASSES_L1 = 49
    # Level 2: Sub-Category
    NUM_CLASSES_L2 = 483
    # Level 3: Fine-Grained Category (Target)
    NUM_CLASSES_L3 = 5270

    # Multi-Task Loss Weights
    WEIGHT_L1 = 0.5
    WEIGHT_L2 = 0.5
    WEIGHT_L3 = 1.0

    # ==========================================
    # TRAINING HYPERPARAMETERS
    # ==========================================
    # Large batch size is feasible since we train MLP on cached features
    BATCH_SIZE = 2048
    EPOCHS = 20
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Regularization
    USE_MIXUP = True
    MIXUP_ALPHA = 0.2
    DROPOUT = 0.5

    # Optimization
    PATIENCE = 5  # For early stopping

    @classmethod
    def setup(cls):
        """
        Creates necessary directories and sets random seeds for reproducibility.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)

        # Set seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
