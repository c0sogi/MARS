import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration for Cassava Leaf Disease Classification.
    Implements parameters for a Hybrid Ensemble (ConvNeXt V2 + Swin V2)
    with Progressive Resizing, LLRD, and SWA.
    """

    # --- General ---
    SEED = 42
    DEBUG = False  # Set to True for fast debugging on a subset
    NUM_CLASSES = 5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # Optimized for the 12 vCPU environment

    # --- Paths ---
    # Input Metadata (Read-only)
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output (Working Directory)
    # Using 'idea_7' as the designated workspace for this run
    OUTPUT_DIR = "./working/idea_7"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    SUBMISSION_FILE = "./submission/submission.csv"
    os.makedirs(os.path.dirname(SUBMISSION_FILE), exist_ok=True)

    # --- Models ---
    # Hybrid Ensemble Architecture
    # 1. ConvNeXt V2 Small: CNN with GRN, initialized with FCMAE weights
    # 2. Swin V2 Tiny: Hierarchical Transformer with window attention
    MODEL_ARCHS = [
        "convnextv2_small.fcmae_ft_in22k_in1k",
        "swinv2_tiny_window16_256.ms_in1k",
    ]

    # --- Data Preprocessing (Progressive Resizing) ---
    IMG_SIZE_LOW = 384  # Phase 1: Coarse training
    IMG_SIZE_HIGH = 512  # Phase 2: Fine-tuning & SWA

    # Normalization (ImageNet defaults)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # Augmentation
    MIXUP_ALPHA = 0.8
    CUTMIX_ALPHA = 1.0
    MIXUP_PROB = 1.0  # Probability of applying MixUp or CutMix

    # Contextual Cropping scale
    CROP_SCALE = (0.3, 1.0)

    # --- Training Hyperparameters ---
    # Batch Size & Gradient Accumulation
    # Target effective batch size >= 32.
    # A100 40GB allows decent batch size even at 512x512.
    BATCH_SIZE = 24
    ACCUM_STEPS = 2  # Effective Batch Size = 48

    # Optimization
    LR_MAX = 2e-4  # Peak learning rate
    WEIGHT_DECAY = 1e-2  # AdamW weight decay
    CLIP_GRAD = 5.0  # Gradient clipping norm

    # Layer-wise Learning Rate Decay (LLRD)
    LLRD_DECAY = 0.8  # Decay factor for deeper layers

    # Regularization
    DROP_PATH_RATE = 0.0  # Disabled for Small/Tiny models with strong augmentation
    DROPOUT_RATE = 0.0  # Standard dropout

    # --- Schedule (SWA Pipeline) ---
    # Total Epochs = Warmup + Base + Fine + SWA
    EPOCHS_WARMUP = 1
    EPOCHS_BASE = 10  # Training at 384x384
    EPOCHS_FINE = 5  # Fine-tuning at 512x512
    EPOCHS_SWA = 5  # SWA at 512x512

    SWA_LR = 5e-5  # Constant/Cyclic LR for SWA phase

    # --- Inference ---
    TTA_FLIPS = True  # Use Horizontal and Vertical flips for TTA


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Apply seeding immediately upon import
seed_everything(Config.SEED)
