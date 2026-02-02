import os
import random
import numpy as np
import torch


class Config:
    # ==========================================
    # DIRECTORIES AND PATHS
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_17"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Input Files
    TRAIN_BSON = os.path.join(INPUT_DIR, "train.bson")
    TEST_BSON = os.path.join(INPUT_DIR, "test.bson")
    CATEGORY_NAMES = os.path.join(INPUT_DIR, "category_names.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # Cache Files (Artifacts)
    HIERARCHY_MAPPING_PATH = os.path.join(WORKING_DIR, "hierarchy_mapping.parquet")

    # Feature Cache Paths
    TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.npy")
    TRAIN_LABELS_PATH = os.path.join(WORKING_DIR, "train_labels.npy")
    VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.npy")
    VAL_LABELS_PATH = os.path.join(WORKING_DIR, "val_labels.npy")
    TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.npy")
    TEST_IDS_PATH = os.path.join(WORKING_DIR, "test_ids.npy")

    # Model Checkpoints & Submission
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "pdfc_model_best.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # ==========================================
    # MODEL ARCHITECTURE
    # ==========================================
    # Dual Backbone Dimensions
    RESNET_DIM = 2048
    EFFICIENTNET_DIM = 1280
    INPUT_DIM = RESNET_DIM + EFFICIENTNET_DIM  # 3328

    # Projected Deep Feature Cascading
    PROJECTION_DIM = 1024
    HIDDEN_DIM = 1024

    # Hierarchy Class Counts
    # Level 1: Coarse categories
    NUM_CLASSES_L1 = 49
    # Level 2: Sub-categories
    NUM_CLASSES_L2 = 483
    # Level 3: Fine-grained targets (Prediction Target)
    NUM_CLASSES_L3 = 5270

    # ==========================================
    # TRAINING CONFIGURATION
    # ==========================================
    SEED = 42

    # Data Loading
    IMG_SIZE = 180
    NUM_WORKERS = 12

    # Training Loop
    BATCH_SIZE = 2048  # Large batch size for MLP training on pre-computed features
    EPOCHS = 20
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_PATIENCE = 5

    # Regularization
    LABEL_SMOOTHING = 0.1
    MIXUP_ALPHA = 0.2

    # Ensemble
    NUM_MODELS = 3

    # Debugging
    DEBUG = False
    DEBUG_SIZE = 10000  # Number of samples to use if DEBUG is True


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
