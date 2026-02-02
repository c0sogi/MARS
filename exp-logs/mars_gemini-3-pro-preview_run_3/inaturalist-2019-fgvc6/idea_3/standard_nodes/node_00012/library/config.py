import os
import torch


class Config:
    """
    Configuration for the iNaturalist Species Classification Task.
    Implements a two-phase training strategy (ConvNeXt-Base).
    """

    # ---------------------------------------------------------
    # General Configuration
    # ---------------------------------------------------------
    SEED = 42
    NUM_CLASSES = 1010
    MODEL_NAME = "convnext_base_in22k"

    # Debugging: Set to True to train on a small subset of data
    DEBUG = False
    DEBUG_SUBSET_SIZE = 5000

    # ---------------------------------------------------------
    # Compute & Hardware
    # ---------------------------------------------------------
    # 12 vCPUs available, so we use a reasonable number of workers
    NUM_WORKERS = 8
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------------------------------------------------
    # Directories & Paths
    # ---------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_3"
    SUBMISSION_DIR = "./submission"

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Paths
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ---------------------------------------------------------
    # Training Phase 1: Representation Learning
    # ---------------------------------------------------------
    # Strategy: Smaller resolution, heavy regularization (Mixup/Cutmix),
    # higher learning rate to learn robust features.
    PHASE_1 = {
        "name": "phase_1",
        "img_size": 224,
        "batch_size": 32,  # Fits in ~16GB VRAM
        "epochs": 20,
        "lr": 1e-4,
        "min_lr": 1e-6,
        "weight_decay": 0.05,
        "warmup_epochs": 3,
        # Regularization
        "mixup_active": True,
        "mixup_alpha": 0.8,
        "cutmix_alpha": 1.0,
        "mixup_prob": 1.0,  # Probability of applying mixup/cutmix
        "label_smoothing": 0.1,  # Applied via SoftTargetCrossEntropy
        "save_path": os.path.join(WORKING_DIR, "phase_1_best.pth"),
    }

    # ---------------------------------------------------------
    # Training Phase 2: Fine-Grained Refinement
    # ---------------------------------------------------------
    # Strategy: Larger resolution, no Mixup (to resolve fine details),
    # lower learning rate, standard CrossEntropy with Label Smoothing.
    PHASE_2 = {
        "name": "phase_2",
        "img_size": 384,
        "batch_size": 16,  # Reduced batch size for larger resolution and 16GB VRAM
        "epochs": 10,
        "lr": 1e-5,  # Lower LR for fine-tuning
        "min_lr": 1e-7,
        "weight_decay": 1e-4,
        "warmup_epochs": 0,
        # Regularization (Mixup Disabled)
        "mixup_active": False,
        "mixup_prob": 0.0,
        "label_smoothing": 0.1,
        "save_path": os.path.join(WORKING_DIR, "phase_2_best.pth"),
    }

    # ---------------------------------------------------------
    # Inference
    # ---------------------------------------------------------
    INFERENCE = {
        "batch_size": 32,
        "tta": True,  # Enable Test Time Augmentation (Horizontal Flip)
        "top_k": 5,  # Number of predictions to save
    }
