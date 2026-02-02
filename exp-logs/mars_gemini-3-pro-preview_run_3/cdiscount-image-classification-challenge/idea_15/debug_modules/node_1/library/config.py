import os
import torch


class Config:
    # ==========================================
    # PATHS & DIRECTORIES
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_15"
    SUBMISSION_DIR = "./submission"

    # Raw Data Sources
    TRAIN_BSON = os.path.join(INPUT_DIR, "train.bson")
    TEST_BSON = os.path.join(INPUT_DIR, "test.bson")
    CATEGORY_NAMES = os.path.join(INPUT_DIR, "category_names.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Splits (Pre-generated)
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # Processed Data Cache (Parquet/NPY)
    # Stores the mapping from category_id to L1/L2/L3 indices
    HIERARCHY_MAPPING_PATH = os.path.join(WORKING_DIR, "hierarchy_mapping.parquet")

    # Feature Cache Paths
    # These store the extracted features from the frozen backbones
    TRAIN_FEATURES = os.path.join(WORKING_DIR, "train_features.npy")
    TRAIN_LABELS = os.path.join(
        WORKING_DIR, "train_labels.npy"
    )  # Stores original category_id
    VAL_FEATURES = os.path.join(WORKING_DIR, "val_features.npy")
    VAL_LABELS = os.path.join(WORKING_DIR, "val_labels.npy")
    TEST_FEATURES = os.path.join(WORKING_DIR, "test_features.npy")
    TEST_IDS = os.path.join(WORKING_DIR, "test_ids.npy")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_CHECKPOINT = os.path.join(WORKING_DIR, "dfc_model_best.pth")

    # ==========================================
    # DATA CONFIGURATION
    # ==========================================
    # Image Processing
    IMG_SIZE = 180  # Native size in BSON
    RESIZE_SIZE = 224  # Input size for Backbones
    MEAN = [0.485, 0.456, 0.406]  # ImageNet Mean
    STD = [0.229, 0.224, 0.225]  # ImageNet Std

    # Hierarchy Dimensions (Based on EDA)
    NUM_CLASSES_L1 = 49
    NUM_CLASSES_L2 = 483
    NUM_CLASSES_L3 = 5270

    # Debugging / Development
    DEBUG = False
    DEBUG_SAMPLES = 5000  # Subset size if DEBUG is True

    # ==========================================
    # MODEL ARCHITECTURE
    # ==========================================
    # Dual Frozen Backbones
    BACKBONE_1 = "resnet50"
    BACKBONE_2 = "efficientnet_b0"

    # Feature Dimensions
    FEAT_DIM_1 = 2048  # ResNet50 output
    FEAT_DIM_2 = 1280  # EfficientNet-B0 output
    INPUT_DIM = FEAT_DIM_1 + FEAT_DIM_2  # 3328 (Concatenated)

    # Cascade Architecture
    HIDDEN_DIM = 1024
    DROPOUT_RATE = 0.3

    # ==========================================
    # TRAINING HYPERPARAMETERS
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 12  # Max available vCPUs

    # Batch Sizes
    EXTRACT_BATCH_SIZE = 512  # Large batch for image feature extraction
    TRAIN_BATCH_SIZE = 4096  # Very large batch for MLP training on features

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 20
    EARLY_STOPPING_PATIENCE = 5

    # Regularization
    LABEL_SMOOTHING = 0.1
    MIXUP_ALPHA = 0.2  # For feature-space MixUp

    @classmethod
    def make_dirs(cls):
        """Creates necessary working and submission directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
