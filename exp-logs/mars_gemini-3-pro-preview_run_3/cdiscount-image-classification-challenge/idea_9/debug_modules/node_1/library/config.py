import os
import torch
import random
import numpy as np


class Config:
    # ==========================================
    # PATH CONFIGURATION
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Idea-specific cache directory
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_9")

    # Submission directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Input Files
    TRAIN_BSON = os.path.join(INPUT_DIR, "train.bson")
    TEST_BSON = os.path.join(INPUT_DIR, "test.bson")
    CATEGORY_NAMES = os.path.join(INPUT_DIR, "category_names.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files (Generated previously)
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # Cached Feature Files
    TRAIN_FEATURES_PATH = os.path.join(CACHE_DIR, "train_features.npy")
    TRAIN_LABELS_PATH = os.path.join(CACHE_DIR, "train_labels.npy")
    VAL_FEATURES_PATH = os.path.join(CACHE_DIR, "val_features.npy")
    VAL_LABELS_PATH = os.path.join(CACHE_DIR, "val_labels.npy")
    TEST_FEATURES_PATH = os.path.join(CACHE_DIR, "test_features.npy")
    TEST_IDS_PATH = os.path.join(CACHE_DIR, "test_ids.npy")

    # Hierarchy Mapping Cache
    HIERARCHY_MAP_PATH = os.path.join(CACHE_DIR, "hierarchy_map.parquet")

    # ==========================================
    # MODEL HYPERPARAMETERS
    # ==========================================
    BACKBONE = "resnet50"
    EMBEDDING_DIM = 2048  # ResNet50 output dim after pooling
    IMG_SIZE = 224

    # Hierarchical Classification Specs
    NUM_CLASSES_L1 = 49
    NUM_CLASSES_L2 = 483
    NUM_CLASSES_L3 = 5270

    # Ensemble Configuration
    ENSEMBLE_SIZE = 5

    # ==========================================
    # TRAINING HYPERPARAMETERS
    # ==========================================
    SEED = 42
    BATCH_SIZE = 2048  # Large batch size for tabular/embedding training
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 15
    PATIENCE = 3  # Early stopping patience

    # MixUp Regularization
    MIXUP_ALPHA = 0.4

    # ==========================================
    # HARDWARE
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Use fewer workers for data loading if doing heavy processing,
    # but for loading embeddings, it's fast.
    NUM_WORKERS = 4

    @staticmethod
    def setup():
        """
        Initializes the environment:
        1. Creates necessary directories.
        2. Sets random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

        # Set seeds
        random.seed(Config.SEED)
        np.random.seed(Config.SEED)
        torch.manual_seed(Config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(Config.SEED)
            # Ensure deterministic behavior where possible
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# Run setup immediately when module is imported to ensure environment is ready
Config.setup()
