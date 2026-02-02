import os
import torch


class Config:
    # -------------------------------------------------------------------------
    # General Configuration
    # -------------------------------------------------------------------------
    SEED = 42
    DEBUG = False
    NUM_WORKERS = 12  # Optimized for 12 vCPUs
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Raw DICOM directories
    TRAIN_DICOM_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DICOM_DIR = os.path.join(INPUT_DIR, "test")

    # Working Directories
    WORKING_DIR = "./working"
    IDEA_DIR = os.path.join(WORKING_DIR, "idea_37")
    CACHE_DIR = IDEA_DIR
    CHECKPOINT_DIR = os.path.join(IDEA_DIR, "checkpoints")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Processing (Fixed Overlapping Orthogonal Tri-Slabs)
    # -------------------------------------------------------------------------
    IMG_SIZE = 240  # Matches EfficientNet-B1 native resolution
    SLAB_COUNT = 3  # 3 slabs per view
    SLAB_OVERLAP = 0.15  # 15% overlap
    VIEWS = ["axial", "coronal"]

    # -------------------------------------------------------------------------
    # Model Architecture (DALA-Net)
    # -------------------------------------------------------------------------
    # Visual Backbone
    BACKBONE_NAME = "efficientnet_b1"
    BACKBONE_PRETRAINED = True
    BACKBONE_DIM = 1280  # Native output dimension (No projection)

    # Tabular Alignment (Deep MLP)
    # Input dim depends on encoding (Age, Sex, Smoking, Percent) -> approx 7-9
    TABULAR_HIDDEN_DIM = 1280  # Matches backbone dim for alignment
    TABULAR_LAYERS = 3

    # Fusion (Lightweight Attention)
    FUSION_HEADS = 4
    FUSION_FFN = False  # Explicitly disabled to prevent overfitting
    FUSION_DROPOUT = 0.1

    # Head (Parametric Inference)
    HEAD_HIDDEN_DIM = 512

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    PATIENCE = 8  # Strict early stopping

    # Scheduler (Cosine Annealing)
    T_MAX = 50
    MIN_LR = 1e-6

    # -------------------------------------------------------------------------
    # Metric / Inference Logic
    # -------------------------------------------------------------------------
    CONFIDENCE_CLIP = 70.0
    MAX_ERROR_CLIP = 1000.0

    @classmethod
    def setup(cls):
        """Creates necessary working directories."""
        for d in [cls.IDEA_DIR, cls.CACHE_DIR, cls.CHECKPOINT_DIR, cls.SUBMISSION_DIR]:
            os.makedirs(d, exist_ok=True)

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print("\n" + "=" * 40)
        print("DALA-Net CONFIGURATION")
        print("=" * 40)
        for k, v in cls.__dict__.items():
            if not k.startswith("__") and not callable(v):
                print(f"{k}: {v}")
        print("=" * 40 + "\n")
