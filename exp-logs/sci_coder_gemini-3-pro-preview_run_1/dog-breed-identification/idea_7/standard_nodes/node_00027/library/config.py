import os
import torch


class Config:
    """
    Configuration for the Dog Breed Classification Task.
    Implements the 'Regularized Heterogeneous Ensemble with Augmentation Annealing' strategy.
    """

    # ==========================================
    # General & Reproducibility
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # File System Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Manifests
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Working & Output Directories
    WORKING_DIR = "./working/idea_7"
    OUTPUT_DIR = WORKING_DIR  # Location to save model checkpoints
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Data Pipeline & Augmentation
    # ==========================================
    # High-Resolution Geometric Pipeline
    RESIZE_DIM = 274
    CROP_SIZE = 256

    # Batch Size (A100 40GB allows for larger batches, but 32 is safe for these large models)
    BATCH_SIZE = 32

    # ==========================================
    # Model Architectures
    # ==========================================
    # Heterogeneous Ensemble: ConvNeXt-Base + Swin-Base
    MODEL_CONFIGS = {
        "convnext_base": {
            "model_name": "convnext_base.fb_in1k",
            "pretrained": True,
            "num_classes": 120,
            "input_size": 256,  # ConvNeXt handles 256x256 natively
            "drop_rate": 0.3,
            "drop_path_rate": 0.2,
        },
        "swin_base": {
            # Swin-Base Window 7 is optimized for 224x224.
            # The training loop should resize the 256x256 crop to 224x224 for this specific model.
            "model_name": "swin_base_patch4_window7_224.ms_in1k",
            "pretrained": True,
            "num_classes": 120,
            "input_size": 224,
            "drop_rate": 0.3,
            "drop_path_rate": 0.2,
        },
    }

    # ==========================================
    # Training Regime
    # ==========================================
    N_FOLDS = 5

    # Four-Phase Training Schedule
    TRAINING_PHASES = {
        # Phase 1: Head Adaptation
        # Freeze backbone, train only the classifier head to align weights.
        "phase_1": {
            "name": "Head Adaptation",
            "epochs": 3,
            "lr": 1e-3,
            "freeze_backbone": True,
            "use_mixup_cutmix": False,
        },
        # Phase 2: Regularized Fine-Tuning
        # Unfreeze backbone, apply discriminative LRs and aggressive regularization (Mixup/CutMix).
        "phase_2": {
            "name": "Regularized Fine-Tuning",
            "epochs": 15,
            "lr_backbone": 5e-5,
            "lr_head": 1e-3,
            "freeze_backbone": False,
            "use_mixup_cutmix": True,
            "mixup_alpha": 0.8,
            "cutmix_alpha": 1.0,
            "mix_prob": 0.5,
        },
        # Phase 3: Regularization Cooldown
        # Disable Mixup/CutMix and lower LR to sharpen predictions and improve calibration.
        "phase_3": {
            "name": "Regularization Cooldown",
            "epochs": 5,
            "lr": 5e-6,
            "freeze_backbone": False,
            "use_mixup_cutmix": False,
        },
        # Phase 4: Stochastic Weight Averaging (SWA)
        # Apply SWA on the cooled model to find the flat minimum.
        "phase_4": {
            "name": "SWA",
            "epochs": 5,
            "lr": 1e-5,
            "swa_start_epoch": 0,  # Apply SWA immediately in this phase
            "use_mixup_cutmix": False,
        },
    }

    # ==========================================
    # Inference
    # ==========================================
    USE_TTA = True  # Enable Test Time Augmentation (Horizontal Flip)
