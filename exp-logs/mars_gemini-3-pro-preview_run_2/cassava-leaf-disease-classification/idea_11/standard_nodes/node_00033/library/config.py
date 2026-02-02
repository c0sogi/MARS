import os
import torch


class Config:
    """
    Configuration module for Cassava Leaf Disease Classification.
    Implements the settings for a Calibrated Progressive Resolution Curriculum
    using ConvNeXt Small with Model EMA and Phase Reset.
    """

    # =========================================================================
    # General Configuration
    # =========================================================================
    SEED = 42
    NUM_CLASSES = 5
    NUM_FOLDS = 5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # 12 vCPUs available
    NUM_WORKERS = 12

    # =========================================================================
    # Directory Paths
    # =========================================================================
    INPUT_DIR = "./input"
    # Metadata files are pre-generated in ./metadata
    METADATA_DIR = "./metadata"
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for this specific experimental idea
    WORKING_DIR = "./working/idea_11"

    # Output sub-directories
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")
    LOG_DIR = os.path.join(WORKING_DIR, "logs")

    # Create directories if they don't exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # =========================================================================
    # Model Architecture
    # =========================================================================
    # ConvNeXt Small (approx 50M params), pre-trained on ImageNet-1k
    MODEL_NAME = "convnext_small.fb_in1k"

    # Regularization: Stochastic Depth (Drop Path)
    DROP_PATH_RATE = 0.4

    # Model EMA (Exponential Moving Average)
    USE_EMA = True
    EMA_DECAY = 0.9999

    # =========================================================================
    # Optimization & Scheduling
    # =========================================================================
    OPTIMIZER_NAME = "AdamW"
    WEIGHT_DECAY = 0.05  # Standard recipe for ConvNeXt
    MAX_GRAD_NORM = 1.0

    # =========================================================================
    # Curriculum Learning Strategy
    # =========================================================================

    # --- Phase 1: Coarse Feature Learning ---
    # Objective: Convergence on global structures with heavy regularization
    PHASE_1 = {
        "image_size": 224,
        "batch_size": 32,
        "epochs": 7,  # ~60% of training budget
        "lr": 2e-4,  # Initial learning rate
        "min_lr": 1e-6,  # Cosine annealing target
        "mixup_prob": 0.5,  # active MixUp/CutMix
        "mixup_alpha": 0.8,
        "cutmix_alpha": 1.0,
        "label_smoothing": 0.0,  # Handled by MixUp
        "use_ema_reset": False,
    }

    # --- Phase 2: Fine-Grained Resolution Tuning ---
    # Objective: Resolve high-freq details, anneal regularization
    PHASE_2 = {
        "image_size": 384,
        "batch_size": 32,  # A100 40GB can handle 32 @ 384x384
        "epochs": 5,  # ~40% of training budget
        "lr": 1e-5,  # Lower LR for fine-tuning
        "min_lr": 1e-7,
        "mixup_prob": 0.0,  # Disable MixUp to see clean pixels
        "label_smoothing": 0.1,  # Use Label Smoothing instead
        "use_ema_reset": True,  # Reset EMA weights to current model weights
    }

    # =========================================================================
    # Inference
    # =========================================================================
    # Test Time Augmentation: Horizontal Flip
    USE_TTA = True

    # =========================================================================
    # Debugging
    # =========================================================================
    # Set to True to run on a small subset of data for pipeline verification
    DEBUG = False
    DEBUG_SUBSET_SIZE = 500
