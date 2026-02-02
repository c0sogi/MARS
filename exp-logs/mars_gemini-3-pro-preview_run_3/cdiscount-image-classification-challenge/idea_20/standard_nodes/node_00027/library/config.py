import os


class Config:
    # ==========================================
    # DIRECTORIES & PATHS
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_20"

    # Create working directory if it doesn't exist
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Raw Data Paths
    TRAIN_BSON = os.path.join(INPUT_DIR, "train.bson")
    TEST_BSON = os.path.join(INPUT_DIR, "test.bson")
    CATEGORY_NAMES = os.path.join(INPUT_DIR, "category_names.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Paths (Pre-generated)
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # Caching Paths (Feature Extraction)
    # We store features as .npy files for fast memory mapping
    TRAIN_FEATURES = os.path.join(WORKING_DIR, "train_features.npy")
    TRAIN_LABELS = os.path.join(WORKING_DIR, "train_labels.npy")

    VAL_FEATURES = os.path.join(WORKING_DIR, "val_features.npy")
    VAL_LABELS = os.path.join(WORKING_DIR, "val_labels.npy")

    TEST_FEATURES = os.path.join(WORKING_DIR, "test_features.npy")
    TEST_IDS = os.path.join(WORKING_DIR, "test_ids.npy")

    # Hierarchy Mapping Cache
    HIERARCHY_MAPPING = os.path.join(WORKING_DIR, "hierarchy_mapping.parquet")

    # Submission Output
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # ==========================================
    # MODEL HYPERPARAMETERS
    # ==========================================
    # Backbone Dimensions
    RESNET_DIM = 2048
    EFFNET_DIM = 1280

    # Projection & Fusion
    PROJECTION_DIM = 1024  # Dimension to project each backbone to
    FUSION_DIM = 2048  # PROJECTION_DIM * 2 (Concatenation)

    # Hierarchical Heads (Class Counts)
    NUM_CLASSES_L1 = 49
    NUM_CLASSES_L2 = 483
    NUM_CLASSES_L3 = 5270  # Target

    # ==========================================
    # TRAINING SETTINGS
    # ==========================================
    SEED = 42
    NUM_WORKERS = 12

    # Feature Extraction
    IMG_SIZE = 224
    EXTRACT_BATCH_SIZE = 256  # Batch size for CNN inference

    # MLP Training
    TRAIN_BATCH_SIZE = 2048  # Large batch size for MLP training on cached features
    EPOCHS = 20  # Sufficient for MLP convergence
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Regularization
    LABEL_SMOOTHING = 0.1
    MIXUP_ALPHA = 0.2
    DROPOUT_RATE = 0.3

    # Ensemble
    NUM_FOLDS = 5  # Number of models in the ensemble

    # Hardware
    DEVICE = "cuda"

    @classmethod
    def print_config(cls):
        print("=" * 30)
        print("CONFIGURATION")
        print("=" * 30)
        print(f"Working Dir: {cls.WORKING_DIR}")
        print(f"Batch Size (Train): {cls.TRAIN_BATCH_SIZE}")
        print(f"Epochs: {cls.EPOCHS}")
        print(f"Ensemble Size: {cls.NUM_FOLDS}")
        print(f"Backbone 1 Dim: {cls.RESNET_DIM}")
        print(f"Backbone 2 Dim: {cls.EFFNET_DIM}")
        print("=" * 30)
