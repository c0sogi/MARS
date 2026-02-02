import os
import torch


class Config:
    """
    Configuration for Tri-Modal Heterogeneous Stacking with Intra-Fold Model Soups.
    """

    # =========================================================================
    # General Setup
    # =========================================================================
    PROJECT_NAME = "DogCat_TriModal_Soup_Idea14"
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # Debugging: Set to True to train on a small subset for pipeline verification
    DEBUG = False
    DEBUG_SUBSET_SIZE = 500

    # =========================================================================
    # Directories
    # =========================================================================
    # Input Directories (Read-Only)
    INPUT_DIR = "./input"
    TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train")
    TEST_IMG_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata Directories (Read-Only)
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Working Directories (Write Allowed)
    # All outputs for this experiment go into idea_14
    WORKING_DIR = "./working/idea_14"
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    OOF_DIR = os.path.join(WORKING_DIR, "oof")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # Create directories immediately upon config import
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OOF_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Data Configuration
    # =========================================================================
    IMG_SIZE = 224
    # Batch size optimized for A100 40GB; adjusted for 'Small' models
    BATCH_SIZE = 64
    N_FOLDS = 5

    # =========================================================================
    # Model Configuration (Tri-Modal Ensemble)
    # =========================================================================
    # 1. ConvNeXt-Small (Pure ConvNet)
    # 2. Swin-Small (Hierarchical Vision Transformer)
    # 3. EfficientNetV2-Small (MBConv / Mobile Inverted Bottleneck)
    MODELS = [
        "convnext_small.fb_in22k",
        "swin_small_patch4_window7_224.ms_in22k",
        "tf_efficientnetv2_s.in21k",
    ]

    # =========================================================================
    # Training Configuration
    # =========================================================================
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4  # Standard for AdamW
    MIN_LR = 1e-6

    # Scheduler: Cosine Annealing
    SCHEDULER_T_MAX = EPOCHS

    # Early Stopping: Disabled (Patience >= Epochs) to allow full convergence
    PATIENCE = 20

    # Intra-Fold Model Soup
    # We will average weights from these epochs (1-based indexing)
    # Corresponds to the end of training where loss landscape is flat
    SOUP_EPOCHS = [18, 19, 20]

    # =========================================================================
    # Augmentation Configuration
    # =========================================================================
    # RandomResizedCrop: Strict minimum scale to preserve subject semantics
    RRC_SCALE_MIN = 0.8
    RRC_SCALE_MAX = 1.0

    # Regularization: Mixup and CutMix
    MIXUP_ALPHA = 0.2
    CUTMIX_ALPHA = 1.0
    # Probability of applying Mixup/CutMix batch-wise
    MIXUP_PROB = 1.0

    # Horizontal Flip (Training & TTA)
    HFLIP_PROB = 0.5

    # =========================================================================
    # Meta-Learner Configuration
    # =========================================================================
    META_MODEL = "LogisticRegression"
