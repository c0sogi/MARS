import os
import torch


class Config:
    # ==========================================
    # 1. Directory & File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific idea (PAVE-Net)
    WORKING_DIR = "./working/idea_39"

    # Cache directory for processed images/features
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Checkpoint directory for model weights
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")

    # Output submission file
    SUBMISSION_FILE = "./submission/submission.csv"

    # ==========================================
    # 2. Data & Preprocessing
    # ==========================================
    SEED = 42
    IMG_SIZE = 224  # Native resolution for EfficientNet-B0
    SLAB_COUNT = 3  # Tri-Slab design
    OVERLAP = 0.15  # 15% overlap between slabs

    # Tabular Features for Progressive Alignment
    # Assumes One-Hot Encoding for Sex (2) and SmokingStatus (3) + Age (1) + Percent (1) = 7 inputs
    TABULAR_COLS = ["Age", "Sex", "SmokingStatus", "Percent"]

    # Features for the Prior-Anchored Skip Connection
    ANCHOR_COLS = ["Baseline_FVC", "Baseline_Percent"]

    # ==========================================
    # 3. Model Architecture (PAVE-Net)
    # ==========================================
    BACKBONE_NAME = "efficientnet-b0"
    BACKBONE_PRETRAINED = True
    BACKBONE_DIM = 1280  # Native output dim of EfficientNet-B0 (No projection)

    # Progressive Tabular Alignment Dimensions
    # Input (7) -> 64 -> 256 -> 1280 (Aligned with visual backbone)
    TABULAR_INPUT_DIM = 7
    TABULAR_HIDDEN_DIMS = [64, 256, 1280]

    # Attention Mechanism
    ATTN_HEADS = 4
    ATTN_DROPOUT = 0.1

    # Feed-Forward Network (FFN) settings
    FFN_DROPOUT = 0.5  # High dropout for non-linear capacity

    # ==========================================
    # 4. Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    EPOCHS = 50
    LR = 1e-4
    WEIGHT_DECAY = 1e-2
    PATIENCE = 8  # Strict early stopping
    NUM_WORKERS = 4

    # Metric Constraints
    MAX_ERROR = 1000  # Metric clipping threshold
    MIN_CONFIDENCE = 70  # Confidence clipping threshold

    # ==========================================
    # 5. System
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Creates the necessary directory structure for the pipeline.
        Should be called at the start of execution.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(cls.SUBMISSION_FILE), exist_ok=True)
