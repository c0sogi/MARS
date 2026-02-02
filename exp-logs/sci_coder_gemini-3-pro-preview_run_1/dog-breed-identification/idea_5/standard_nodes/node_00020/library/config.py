import os
import torch


class Config:
    """
    Configuration class for Dog Breed Classification Task.
    Implements settings for a Heterogeneous Ensemble (ConvNeXt + Swin Transformer).
    """

    # -------------------------------------------------------------------------
    # General Setup
    # -------------------------------------------------------------------------
    SEED = 42
    NUM_FOLDS = 5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use available CPUs for data loading, capped reasonably
    NUM_WORKERS = min(os.cpu_count(), 12)

    # -------------------------------------------------------------------------
    # Directory Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_5"

    # Ensure working directory exists for caching and checkpoints
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Submission Output Path
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Preprocessing & Loading
    # -------------------------------------------------------------------------
    NUM_CLASSES = 120

    # Image sizing strategy: Resize short edge to 256, then CenterCrop to 224
    # This preserves aspect ratio better than resizing directly to (224, 224)
    RESIZE_SIZE = 256
    IMAGE_SIZE = 224

    BATCH_SIZE = 32  # Conservative batch size for A100 to handle Base models safely

    # -------------------------------------------------------------------------
    # Model Architectures
    # -------------------------------------------------------------------------
    # Using timm model names.
    # 1. ConvNeXt Base: Strong CNN baseline, good for local features/textures.
    # 2. Swin Transformer Base: Hierarchical Vision Transformer, good for global context.
    # Both initialized with ImageNet-1k weights (avoiding 22k to prevent distribution shift issues).
    MODEL_ARCHS = ["convnext_base.fb_in1k", "swin_base_patch4_window7_224.ms_in1k"]

    # -------------------------------------------------------------------------
    # Training Hyperparameters (Two-Phase Strategy)
    # -------------------------------------------------------------------------
    # General Optimization
    WEIGHT_DECAY = 0.01  # Standard weight decay for AdamW on ViT/ConvNeXt

    # Phase 1: Head Adaptation
    # Freeze backbone, train only the classifier head to align weights.
    EPOCHS_HEAD = 3
    LR_HEAD_INIT = 1e-3

    # Phase 2: Full Fine-Tuning
    # Unfreeze backbone, use discriminative learning rates.
    EPOCHS_FINE = 12
    LR_BACKBONE = 1e-6  # Very low rate for backbone to preserve pre-trained features
    LR_HEAD_FINE = 1e-4  # Moderate rate for head to learn specific breed distinctions

    # -------------------------------------------------------------------------
    # Inference Strategy
    # -------------------------------------------------------------------------
    USE_TTA = True  # Enable Test Time Augmentation (Horizontal Flip)
