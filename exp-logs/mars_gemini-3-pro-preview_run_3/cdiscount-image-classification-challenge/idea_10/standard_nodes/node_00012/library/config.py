import os
import torch
import numpy as np
import random


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    # ==========================================
    # GLOBAL SETUP
    # ==========================================
    SEED = 42

    # Debugging / Development Flags
    # Set DEBUG to True to run on a small subset of data for testing the pipeline
    DEBUG = False
    DEBUG_SUBSET_SIZE = 50000

    # Hardware
    NUM_WORKERS = 12
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # DIRECTORIES & PATHS
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_10"

    # Ensure the working directory exists
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

    # Cache Paths (For Decoupled Feature Extraction)
    # These files will store the extracted features to disk to save RAM/Time
    HIERARCHY_MAPPING_PATH = os.path.join(WORKING_DIR, "hierarchy_mapping.parquet")

    TRAIN_FEATURES = os.path.join(WORKING_DIR, "train_features.npy")
    TRAIN_LABELS = os.path.join(WORKING_DIR, "train_labels.npy")

    VAL_FEATURES = os.path.join(WORKING_DIR, "val_features.npy")
    VAL_LABELS = os.path.join(WORKING_DIR, "val_labels.npy")

    TEST_FEATURES = os.path.join(WORKING_DIR, "test_features.npy")
    TEST_IDS = os.path.join(WORKING_DIR, "test_ids.npy")

    # Output Paths
    MODEL_CHECKPOINT = os.path.join(WORKING_DIR, "hierarchical_mlp_best.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # ==========================================
    # MODEL ARCHITECTURE
    # ==========================================
    # Feature Extraction
    # We use a concatenation of ResNet50 and EfficientNet-B0 features
    FEAT_DIM_RESNET = 2048
    FEAT_DIM_EFFNET = 1280
    INPUT_DIM = FEAT_DIM_RESNET + FEAT_DIM_EFFNET  # Total: 3328

    # MLP Head Architecture
    HIDDEN_LAYERS = [2048, 1024]
    DROPOUT_RATE = 0.3

    # Hierarchical Targets
    # Level 1: Coarse Categories (e.g., SPORT)
    # Level 2: Sub-Categories (e.g., CYCLES)
    # Level 3: Fine-Grained (Target) (e.g., VELO VILLE)
    NUM_CLASSES_L1 = 49
    NUM_CLASSES_L2 = 483
    NUM_CLASSES_L3 = 5270

    # ==========================================
    # TRAINING HYPERPARAMETERS
    # ==========================================
    # Since we are training on pre-extracted features (vectors), we can use a large batch size
    BATCH_SIZE = 2048

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 30
    EARLY_STOPPING_PATIENCE = 5

    # Regularization
    LABEL_SMOOTHING = 0.1
    MIXUP_ALPHA = 0.2  # Alpha for Beta distribution in MixUp


# Apply seeding immediately upon import
seed_everything(Config.SEED)
