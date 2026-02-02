import os


class Config:
    """
    Global configuration for the Product Categorization task.
    Implements the settings for 'Full-Scale Dual-Backbone Fusion with Hierarchical Ensemble'.
    """

    # ==========================================
    # DIRECTORIES
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Idea-specific cache directory
    IDEA_ID = "idea_13"
    CACHE_DIR = os.path.join(WORKING_DIR, IDEA_ID)
    MODEL_DIR = os.path.join(CACHE_DIR, "models")

    # ==========================================
    # INPUT FILE PATHS
    # ==========================================
    # Raw BSON Data
    TRAIN_BSON = os.path.join(INPUT_DIR, "train.bson")
    TEST_BSON = os.path.join(INPUT_DIR, "test.bson")

    # Auxiliary Data
    CATEGORY_NAMES = os.path.join(INPUT_DIR, "category_names.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Pre-computed Metadata (Indices)
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # ==========================================
    # CACHED ARTIFACT PATHS
    # ==========================================
    # Feature Caches (Numpy Memory-Mapped compatible)
    # Storing fused embeddings (ResNet + EfficientNet)
    TRAIN_FEATURES = os.path.join(CACHE_DIR, "train_features.npy")
    TRAIN_LABELS = os.path.join(CACHE_DIR, "train_labels.npy")  # Level 3 (Target)

    VAL_FEATURES = os.path.join(CACHE_DIR, "val_features.npy")
    VAL_LABELS = os.path.join(CACHE_DIR, "val_labels.npy")  # Level 3 (Target)

    TEST_FEATURES = os.path.join(CACHE_DIR, "test_features.npy")
    TEST_IDS = os.path.join(CACHE_DIR, "test_ids.npy")  # Product IDs for submission

    # Hierarchy Data
    # Parquet file to map category_id (L3) -> L2 -> L1
    HIERARCHY_MAPPING = os.path.join(CACHE_DIR, "hierarchy_mapping.parquet")

    # Final Output
    SUBMISSION_PATH = os.path.join(CACHE_DIR, "submission.csv")

    # ==========================================
    # HYPERPARAMETERS
    # ==========================================
    SEED = 42

    # --- Feature Extraction ---
    IMG_SIZE = 224  # Standard input size for ResNet/EfficientNet
    EXTRACT_BATCH_SIZE = 256  # High batch size for A100 inference
    NUM_WORKERS = 12  # Maximize CPU usage for BSON decoding

    # Feature Dimensions
    RESNET_DIM = 2048
    EFFNET_DIM = 1280
    TOTAL_FEAT_DIM = RESNET_DIM + EFFNET_DIM  # 3328

    # --- Training (MLP Ensemble) ---
    TRAIN_BATCH_SIZE = (
        4096  # Large batch size for MLP training on pre-computed features
    )
    EPOCHS = 30
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 5  # Early stopping patience

    # Regularization
    MIXUP_ALPHA = 0.2  # Alpha for feature-space MixUp
    DROPOUT_RATE = 0.3

    # Ensemble Strategy
    ENSEMBLE_SIZE = 5

    # --- Hierarchy Structure ---
    # Derived from dataset analysis
    NUM_CLASSES_L1 = 49
    NUM_CLASSES_L2 = 483
    NUM_CLASSES_L3 = 5270

    @classmethod
    def setup(cls):
        """Creates necessary working directories."""
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
