import os
import torch


class Config:
    """
    Global configuration for the Product Categorization Task.
    Implements the 'Full-Scale Dual-Backbone with Projected Multi-Task Learning (PMTL)' strategy.
    """

    # ==========================================
    # SYSTEM & HARDWARE
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 12  # Utilizing available vCPUs

    # ==========================================
    # PATHS
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_18"
    SUBMISSION_DIR = "./submission"

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Source Data
    TRAIN_BSON = os.path.join(INPUT_DIR, "train.bson")
    TEST_BSON = os.path.join(INPUT_DIR, "test.bson")
    CATEGORY_NAMES = os.path.join(INPUT_DIR, "category_names.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # Decoupled Feature Cache (Parquet/NPY)
    # These files store the pre-computed features from the frozen backbones
    TRAIN_FEATURES = os.path.join(WORKING_DIR, "train_features.npy")
    TRAIN_LABELS = os.path.join(WORKING_DIR, "train_labels.npy")

    VAL_FEATURES = os.path.join(WORKING_DIR, "val_features.npy")
    VAL_LABELS = os.path.join(WORKING_DIR, "val_labels.npy")

    TEST_FEATURES = os.path.join(WORKING_DIR, "test_features.npy")
    TEST_IDS = os.path.join(WORKING_DIR, "test_ids.npy")

    # Auxiliary Data
    HIERARCHY_MAPPING = os.path.join(WORKING_DIR, "hierarchy_mapping.parquet")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # DATA PROCESSING & FEATURE EXTRACTION
    # ==========================================
    IMG_SIZE = 224
    EXTRACT_BATCH_SIZE = 256  # Batch size for CNN forward pass

    # Debugging flags to run on a subset of data
    DEBUG = False
    DEBUG_SAMPLES = 10000

    # ==========================================
    # MODEL ARCHITECTURE
    # ==========================================
    # Dual Frozen Backbones
    BACKBONES = ["resnet50", "efficientnet_b0"]

    # Feature Dimensions
    # ResNet50 (2048) + EfficientNet-B0 (1280) = 3328
    BACKBONE_DIMS = [2048, 1280]
    INPUT_DIM = sum(BACKBONE_DIMS)

    # Projection Layer (Bottleneck)
    PROJECTION_DIM = 1024

    # Hierarchical Heads
    NUM_CLASSES_L1 = 49
    NUM_CLASSES_L2 = 483
    NUM_CLASSES_L3 = 5270  # Target

    DROPOUT_RATE = 0.3

    # ==========================================
    # TRAINING HYPERPARAMETERS
    # ==========================================
    # Training the MLP on top of cached features allows for very large batch sizes
    TRAIN_BATCH_SIZE = 4096
    EPOCHS = 30

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Regularization
    LABEL_SMOOTHING = 0.1
    MIXUP_ALPHA = 0.2  # Feature-space MixUp

    # Early Stopping
    PATIENCE = 5

    # Ensemble Strategy
    NUM_MODELS = 5  # Number of independent models to train for ensemble
