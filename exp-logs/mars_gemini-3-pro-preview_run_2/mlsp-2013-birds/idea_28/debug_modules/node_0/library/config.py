import os
import torch


class Config:
    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific idea/experiment
    WORKING_DIR = "./working/idea_28"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Sub-directories for storing intermediate distillation targets (OOF predictions)
    GEN0_DIR = os.path.join(WORKING_DIR, "gen0_anchors")
    GEN1_DIR = os.path.join(WORKING_DIR, "gen1_stabilization")
    GEN2_DIR = os.path.join(WORKING_DIR, "gen2_refinement")

    SUBMISSION_DIR = "./submission"

    # Data Sources
    SPECTROGRAM_DIR = os.path.join(
        INPUT_DIR, "supplemental_data", "filtered_spectrograms"
    )
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # =========================================================================
    # Data Configuration
    # =========================================================================
    SEED = 42
    NUM_CLASSES = 19

    # Input Dimensions: 224 (Freq) x 448 (Time)
    # 1:2 Aspect Ratio preserves temporal fidelity
    IMG_HEIGHT = 224
    IMG_WIDTH = 448

    # Normalization (ImageNet stats for Pseudo-RGB)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # =========================================================================
    # Model Architecture
    # =========================================================================
    # Heterogeneous Ensemble Components
    MODEL_RESNET = "resnet18"
    MODEL_EFFICIENTNET = "efficientnet_b0"
    MODEL_DENSENET = "densenet121"

    # Generation 0: Stable Anchors
    ANCHOR_MODELS = [MODEL_RESNET, MODEL_EFFICIENTNET]

    # Generation 1 & 2: Full Ensemble
    FULL_ENSEMBLE_MODELS = [MODEL_RESNET, MODEL_EFFICIENTNET, MODEL_DENSENET]

    # Multi-Sample Dropout Rates for Head
    DROPOUT_RATES = [0.1, 0.2, 0.3, 0.4, 0.5]

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    # Hardware
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Optimization
    EPOCHS = 25  # Per generation
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Augmentation
    MIXUP_ALPHA = 0.4

    # Distillation
    # Loss = BCE(GroundTruth) + LAMBDA * BCE(SoftTargets)
    DISTILLATION_LAMBDA = 1.0

    # Test-Time Augmentation (Cyclic Time-Rolling)
    # Shifts as fraction of image width
    TTA_SHIFTS = [0.0, 0.25, 0.50, 0.75]

    # Cross-Validation
    NUM_FOLDS = 5

    # =========================================================================
    # Debugging & Control
    # =========================================================================
    DEBUG = False
    DEBUG_SUBSET_SIZE = 50  # Number of samples to use if DEBUG is True

    @classmethod
    def setup(cls):
        """Creates necessary working directories."""
        dirs = [
            cls.WORKING_DIR,
            cls.CACHE_DIR,
            cls.GEN0_DIR,
            cls.GEN1_DIR,
            cls.GEN2_DIR,
            cls.SUBMISSION_DIR,
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

        # Set deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
