import os
import torch


class Config:
    """
    Configuration for the Translation-Invariant SegFormer (MiT-B2)
    with Constrained Dynamic Sampling.
    """

    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching intermediate files and model checkpoints
    # Specific to Idea 27
    WORKING_DIR = "./working/idea_27"
    CACHE_DIR = WORKING_DIR

    # Final submission file location
    SUBMISSION_PATH = "./submission.csv"

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # =========================================================================
    # Data Preprocessing & Z-Slice Strategy
    # =========================================================================
    # Input Image Dimensions
    TILE_SIZE = 512

    # "Overlapping Thick Slab" Configuration
    # We take a slab of 12 slices and split it into 3 channels via MIP (Max Intensity Projection).
    # Channel 1: MIP(z, z+4)
    # Channel 2: MIP(z+4, z+8)
    # Channel 3: MIP(z+8, z+12)
    SLAB_DEPTH = 12
    NUM_CHANNELS = 3
    SLICES_PER_CHANNEL = 4  # SLAB_DEPTH // NUM_CHANNELS

    # Constrained Dynamic Sampling (Training)
    # The start index of the slab is randomly sampled from this range.
    # This ensures the central ink volume (approx slice 32) is always captured
    # but shifts between channels, forcing translation invariance.
    TRAIN_Z_MIN = 16
    TRAIN_Z_MAX = 24

    # Deterministic Z-Scanning (Inference)
    # We generate predictions for these fixed start indices and Max-Fuse them.
    INFERENCE_Z_STARTS = [16, 20, 24]

    # Normalization (Standard ImageNet Statistics)
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]

    # =========================================================================
    # Model Architecture
    # =========================================================================
    # SegFormer MiT-B2 with MLP Decoder
    ENCODER_NAME = "mit_b2"
    ENCODER_WEIGHTS = "imagenet"
    DECODER_DIM = 256  # Standard embedding dim for SegFormer MLP head
    CLASSES = 1  # Binary classification (Ink vs No-Ink)

    # =========================================================================
    # Training Hyperparameters (Micro-Dataset Protocol)
    # =========================================================================
    BATCH_SIZE = 8
    LEARNING_RATE = 6e-5
    NUM_EPOCHS = 20  # Sufficient for convergence with early stopping

    # Optimizer Settings
    WEIGHT_DECAY = 1e-2
    OPTIMIZER = "AdamW"

    # Loss Function Weights
    BCE_WEIGHT = 0.5
    DICE_WEIGHT = 0.5

    # Hardware & Reproducibility
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    SEED = 42

    # =========================================================================
    # Validation & Inference
    # =========================================================================
    # Threshold for converting probability maps to binary masks
    BINARIZATION_THRESHOLD = 0.5

    # Beta value for F-score (F0.5 weights precision higher than recall)
    F_BETA = 0.5

    # Baseline score to beat for saving checkpoints
    PREV_BEST_SCORE = 0.598

    @classmethod
    def setup(cls):
        """
        Initialize the environment by creating necessary directories.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        print(f"Configured Working Directory: {cls.WORKING_DIR}")
        print(f"Device: {cls.DEVICE}")
