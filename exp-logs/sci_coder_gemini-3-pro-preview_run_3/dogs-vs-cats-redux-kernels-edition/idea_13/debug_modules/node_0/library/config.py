import os
import torch


class Config:
    """
    Configuration for the Heterogeneous Ensemble with SSL-ViT.
    """

    # -------------------------------------------------------------------------
    # Global Constants & Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42
    NUM_CLASSES = 1  # Binary classification (Dog vs Cat)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------------------------------
    # Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_13"
    SUBMISSION_DIR = "./submission"

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Data Pipeline
    # -------------------------------------------------------------------------
    NUM_WORKERS = 4

    # Context-Preserving Augmentation
    # Scale (0.8, 1.0) ensures the subject is not cropped out
    AUG_CROP_SCALE = (0.8, 1.0)
    # Color Jitter Intensity >= 0.2
    AUG_COLOR_JITTER = 0.2

    # -------------------------------------------------------------------------
    # Model Configurations
    # -------------------------------------------------------------------------
    MODEL_CONFIGS = {
        "resnet": {
            # ResNet-50 (Standard CNN) - Supervised
            # Using .tv_in1k for standard torchvision weights (often updated to V2 recipes in modern timm)
            "model_name": "resnet50.tv_in1k",
            "img_size": 256,
            "batch_size": 64,
            "learning_rate": 1e-4,
            "epochs": 6,
            "use_llrd": False,
            "llrd_decay": 0.9,
            "weight_decay": 1e-4,
            "min_lr": 1e-6,
        },
        "convnext": {
            # ConvNeXt-Small (Modern CNN) - Supervised
            "model_name": "convnext_small.fb_in1k",
            "img_size": 256,
            "batch_size": 64,
            "learning_rate": 1e-4,
            "epochs": 6,
            "use_llrd": False,
            "llrd_decay": 0.9,
            "weight_decay": 1e-4,
            "min_lr": 1e-6,
        },
        "vit_ssl": {
            # ViT-Small (Transformer) - Self-Supervised (DINO)
            # Using vit_small_patch16_224.dino to match Patch16-224 spec and SSL requirement
            "model_name": "vit_small_patch16_224.dino",
            "img_size": 224,  # Native resolution for ViT
            "batch_size": 64,
            "learning_rate": 5e-5,  # Lower LR for fine-tuning SSL weights
            "epochs": 6,
            "use_llrd": True,  # Layer-wise Learning Rate Decay
            "llrd_decay": 0.9,
            "weight_decay": 0.05,
            "min_lr": 1e-6,
        },
    }

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------
    # Test Time Augmentation: Horizontal Flip
    TTA_FLIP = True
