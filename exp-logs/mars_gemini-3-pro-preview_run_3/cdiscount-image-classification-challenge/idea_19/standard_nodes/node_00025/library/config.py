import os
import torch
import numpy as np
import random


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Config:
    # ==========================================
    # REPRODUCIBILITY
    # ==========================================
    SEED = 42

    # ==========================================
    # PATHS
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_19"
    SUBMISSION_DIR = "./submission"

    # Raw Data Sources
    TRAIN_BSON = os.path.join(INPUT_DIR, "train.bson")
    TEST_BSON = os.path.join(INPUT_DIR, "test.bson")
    CATEGORY_NAMES = os.path.join(INPUT_DIR, "category_names.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files (Generated previously)
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # Feature Cache Paths (NPY format for memory mapping/fast I/O)
    # Dual-Stream: ResNet50 + EfficientNetB0
    TRAIN_FEATS_RESNET = os.path.join(WORKING_DIR, "train_feats_resnet.npy")
    TRAIN_FEATS_EFFNET = os.path.join(WORKING_DIR, "train_feats_effnet.npy")
    TRAIN_LABELS = os.path.join(WORKING_DIR, "train_labels.npy")  # Stores [L1, L2, L3]

    VAL_FEATS_RESNET = os.path.join(WORKING_DIR, "val_feats_resnet.npy")
    VAL_FEATS_EFFNET = os.path.join(WORKING_DIR, "val_feats_effnet.npy")
    VAL_LABELS = os.path.join(WORKING_DIR, "val_labels.npy")

    TEST_FEATS_RESNET = os.path.join(WORKING_DIR, "test_feats_resnet.npy")
    TEST_FEATS_EFFNET = os.path.join(WORKING_DIR, "test_feats_effnet.npy")
    TEST_IDS = os.path.join(WORKING_DIR, "test_ids.npy")

    # Auxiliary Mappings
    HIERARCHY_MAPPING = os.path.join(WORKING_DIR, "hierarchy_mapping.parquet")

    # Model Artifacts
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "dual_stream_model_best.pth")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # DATA PROCESSING & DEBUGGING
    # ==========================================
    IMG_SIZE = 224

    # Debugging / Development Flags
    # Set DEBUG_SIZE to a small integer (e.g., 10000) to run a quick end-to-end test
    DEBUG_SIZE = None

    # ==========================================
    # MODEL ARCHITECTURE
    # ==========================================
    # Input Dimensions from Backbones
    RESNET_DIM = 2048
    EFFNET_DIM = 1280

    # Projection & Fusion
    PROJECTION_DIM = 1024
    FUSION_DIM = 2048  # (1024 from ResNet stream + 1024 from EffNet stream)

    # Output Dimensions (Hierarchical Levels)
    NUM_CLASSES_L1 = 49
    NUM_CLASSES_L2 = 483
    NUM_CLASSES_L3 = 5270

    # ==========================================
    # TRAINING HYPERPARAMETERS
    # ==========================================
    BATCH_SIZE = 4096  # Large batch size for MLP training on cached features
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 20
    EARLY_STOPPING_PATIENCE = 3

    # Regularization
    LABEL_SMOOTHING = 0.1
    MIXUP_ALPHA = 0.2
    DROPOUT_RATE = 0.3

    # ==========================================
    # HARDWARE & SYSTEM
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 12  # Optimal for the given 12 vCPU environment

    @classmethod
    def setup(cls):
        """
        Initialize the working environment:
        1. Create necessary directories.
        2. Set random seeds for reproducibility.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        set_seed(cls.SEED)
