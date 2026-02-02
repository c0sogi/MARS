import os
import torch


class Config:
    """
    Global configuration for the Deep Feature Cascading (DFC) solution.
    Handles paths, hyperparameters, and hardware settings.
    """

    # ==========================================
    # REPRODUCIBILITY & DEBUGGING
    # ==========================================
    SEED = 42

    # Debug flag: If True, pipeline runs on a small subset of data
    DEBUG = False
    DEBUG_SUBSET_SIZE = 10000

    # ==========================================
    # DIRECTORY PATHS
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Solution-specific working directory
    IDEA_DIR = os.path.join(WORKING_DIR, "idea_16")
    MODEL_DIR = os.path.join(IDEA_DIR, "models")

    # ==========================================
    # INPUT DATA PATHS
    # ==========================================
    # Raw BSON files
    TRAIN_BSON_PATH = os.path.join(INPUT_DIR, "train.bson")
    TEST_BSON_PATH = os.path.join(INPUT_DIR, "test.bson")

    # Auxiliary files
    CATEGORY_NAMES_PATH = os.path.join(INPUT_DIR, "category_names.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Pre-computed Metadata (CSV)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # ==========================================
    # CACHED ARTIFACTS (INTERMEDIATE DATA)
    # ==========================================
    # We use .npy files for efficient memory-mapped loading of large feature matrices
    TRAIN_FEATURES_PATH = os.path.join(IDEA_DIR, "train_features.npy")
    TRAIN_LABELS_PATH = os.path.join(IDEA_DIR, "train_labels.npy")

    VAL_FEATURES_PATH = os.path.join(IDEA_DIR, "val_features.npy")
    VAL_LABELS_PATH = os.path.join(IDEA_DIR, "val_labels.npy")

    TEST_FEATURES_PATH = os.path.join(IDEA_DIR, "test_features.npy")
    TEST_IDS_PATH = os.path.join(IDEA_DIR, "test_ids.npy")

    # Hierarchy mapping (Category ID -> Level 1, 2, 3)
    HIERARCHY_MAPPING_PATH = os.path.join(IDEA_DIR, "hierarchy_mapping.parquet")

    # Final Submission Output
    SUBMISSION_PATH = os.path.join(IDEA_DIR, "submission.csv")

    # ==========================================
    # COMPUTE RESOURCES
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # 12 vCPUs available, leaving some overhead for system processes
    NUM_WORKERS = 8
    PIN_MEMORY = True

    # ==========================================
    # FEATURE EXTRACTION CONFIG
    # ==========================================
    # Image preprocessing
    IMG_SIZE = 224  # Standard input size for ResNet/EfficientNet
    IMG_MEAN = [0.485, 0.456, 0.406]  # ImageNet statistics
    IMG_STD = [0.229, 0.224, 0.225]

    # Extraction Batch Size (A100 40GB can handle large batches)
    BATCH_SIZE_EXTRACTION = 256

    # Backbone Dimensions (Frozen)
    DIM_RESNET = 2048  # ResNet50
    DIM_EFFNET = 1280  # EfficientNet-B0
    DIM_INPUT = DIM_RESNET + DIM_EFFNET  # 3328 Total Input Dimension

    # ==========================================
    # MODEL ARCHITECTURE (DFC)
    # ==========================================
    # Hierarchy Levels (from EDA)
    NUM_CLASSES_L1 = 49
    NUM_CLASSES_L2 = 483
    NUM_CLASSES_L3 = 5270

    # Network Hyperparameters
    HIDDEN_DIM = 1024
    DROPOUT_RATE = 0.3

    # ==========================================
    # TRAINING HYPERPARAMETERS
    # ==========================================
    # MLP training on pre-computed features is memory efficient
    BATCH_SIZE_TRAIN = 4096

    # Optimization
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Regularization
    LABEL_SMOOTHING = 0.1
    MIXUP_ALPHA = 0.2

    # Ensemble Strategy
    NUM_ENSEMBLE_MODELS = 3

    @classmethod
    def setup(cls):
        """
        Initializes the working environment by creating necessary directories.
        """
        os.makedirs(cls.IDEA_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
