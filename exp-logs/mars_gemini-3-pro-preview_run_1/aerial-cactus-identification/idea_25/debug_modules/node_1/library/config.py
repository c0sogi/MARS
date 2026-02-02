import os
import torch


class Config:
    # ==========================================
    # Meta Configuration
    # ==========================================
    SEED = 42
    IDEA_ID = "idea_25"
    DEBUG = False  # Set to True to run on a small subset for debugging

    # ==========================================
    # Directory Setup
    # ==========================================
    # Input (Read-Only)
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working (Read/Write)
    WORKING_DIR = f"./working/{IDEA_ID}"
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Ensure writable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # ==========================================
    # Data Parameters
    # ==========================================
    IMG_SIZE = 32
    NUM_CLASSES = 1
    NUM_FOLDS = 5
    BATCH_SIZE = 128
    NUM_WORKERS = 4

    # Normalization Constants (Calculated from Dataset Analysis)
    # RGB Mean and Std normalized to [0, 1]
    # Mean: R=128.37, G=115.25, B=119.40 -> [0.503, 0.452, 0.468]
    # Std:  R=38.60,  G=35.68,  B=39.15  -> [0.151, 0.140, 0.154]
    NORM_MEAN = [0.503, 0.452, 0.468]
    NORM_STD = [0.151, 0.140, 0.154]

    # ==========================================
    # Model Parameters
    # ==========================================
    MODEL_NAME = "MultiTaskRepVGG"
    # Backbone settings
    BACKBONE_WIDTH_MULTIPLIER = 1.0
    # Head settings
    USE_TEXTURE_HEAD = True
    USE_SEMANTIC_HEAD = True
    USE_QUALITY_HEAD = True  # Auxiliary regression head for file size

    # ==========================================
    # Training / Optimization (SWA Strategy)
    # ==========================================
    # Phase 1: Convergence (AdamW + Cosine Annealing)
    CONVERGENCE_EPOCHS = 25

    # Phase 2: Exploration (SWA with Cyclic LR)
    SWA_EPOCHS = 10
    SWA_START_EPOCH = CONVERGENCE_EPOCHS
    SWA_LR_MAX = 2e-3
    SWA_LR_MIN = 1e-5
    SWA_CYCLE_LEN = 2  # Capture snapshot every 2 epochs

    # Total Training Duration
    TOTAL_EPOCHS = CONVERGENCE_EPOCHS + SWA_EPOCHS

    # Optimizer Settings
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2  # Slightly higher for regularization

    # Regularization
    MIXUP_ALPHA = 0.2

    # Loss Weights for Multi-Task Learning
    # L_total = w_t * L_texture + w_s * L_semantic + w_q * L_quality
    LOSS_WEIGHT_TEXTURE = 1.0
    LOSS_WEIGHT_SEMANTIC = 1.0
    LOSS_WEIGHT_QUALITY = 0.5  # Lambda for regression head (MSE)

    # ==========================================
    # Hardware
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def print_config(cls):
        print(f"Configuration for {cls.IDEA_ID}:")
        print(f"  Device: {cls.DEVICE}")
        print(f"  Image Size: {cls.IMG_SIZE}x{cls.IMG_SIZE}")
        print(f"  Batch Size: {cls.BATCH_SIZE}")
        print(f"  Folds: {cls.NUM_FOLDS}")
        print(
            f"  Epochs: {cls.CONVERGENCE_EPOCHS} (Conv) + {cls.SWA_EPOCHS} (SWA) = {cls.TOTAL_EPOCHS}"
        )
        print(f"  Mixup Alpha: {cls.MIXUP_ALPHA}")
        print(
            f"  Loss Weights: Texture={cls.LOSS_WEIGHT_TEXTURE}, Semantic={cls.LOSS_WEIGHT_SEMANTIC}, Quality={cls.LOSS_WEIGHT_QUALITY}"
        )
        print(f"  Cache Dir: {cls.CACHE_DIR}")
