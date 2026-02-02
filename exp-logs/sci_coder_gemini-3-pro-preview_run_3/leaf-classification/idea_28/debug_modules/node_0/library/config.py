import os
import random
import numpy as np
import torch


class Config:
    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Specific cache directory for this solution (Idea 28)
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_28")

    # Submission output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Input sub-paths
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # ==========================================
    # Global Hyperparameters
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to limit dataset size for debugging
    MAX_DEBUG_SAMPLES = 50  # Only used if DEBUG is True

    # Compute
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    BATCH_SIZE = 32

    # ==========================================
    # Data Processing Parameters
    # ==========================================
    # Image preprocessing
    IMG_SIZE = 224  # Standard input size for DINOv2 and ConvNeXt

    # Manifold Densification Strategy
    N_ROTATIONS = 12  # Number of equidistant rotations (0, 30, ..., 330)
    N_CENTROIDS = 3  # Number of orthogonal centroids to generate per image

    # Centroid definitions (indices of the 12 rotations)
    # 12 rotations correspond to indices 0..11
    # Centroid A: {0, 90, 180, 270} -> Indices [0, 3, 6, 9]
    # Centroid B: {30, 120, 210, 300} -> Indices [1, 4, 7, 10]
    # Centroid C: {60, 150, 240, 330} -> Indices [2, 5, 8, 11]
    CENTROID_INDICES = [
        [0, 3, 6, 9],  # Centroid A
        [1, 4, 7, 10],  # Centroid B
        [2, 5, 8, 11],  # Centroid C
    ]

    # ==========================================
    # Model Architecture Parameters
    # ==========================================
    # Feature Extractors
    # Using timm model names
    DINO_MODEL_NAME = "vit_large_patch14_dinov2.lvd142m"
    CONVNEXT_MODEL_NAME = "convnext_large.fb_in22k_ft_in1k"

    # Dimensionality Reduction
    PCA_VARIANCE = 0.99  # Retain 99% variance for visual streams

    # Tabular Processing
    TABULAR_TRANSFORMER_OUTPUT = "normal"  # For QuantileTransformer

    # Classifier
    CLASSIFIER_SOLVER = "lsqr"
    CLASSIFIER_SHRINKAGE = "auto"  # Ledoit-Wolf shrinkage

    # ==========================================
    # Training Parameters
    # ==========================================
    N_FOLDS = 10


def setup_system(seed=Config.SEED):
    """
    Sets up the environment:
    1. Creates necessary directories.
    2. Sets random seeds for reproducibility.
    """
    # Create directories
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(
        f"System setup complete. Cache: {Config.CACHE_DIR}, Device: {Config.DEVICE}, Seed: {seed}"
    )
