import os
import torch
import numpy as np


class Config:
    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata Paths (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Working Directory for Caching (Idea 9)
    WORKING_DIR = "./working/idea_9"
    CACHE_DIR = WORKING_DIR

    # Output Directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Data Processing Parameters
    # ==========================================
    # Image Dimensions for EfficientNet-B0
    IMG_SIZE = 224

    # Hounsfield Unit (HU) Thresholds for Lung Masking
    HU_MIN = -1000
    HU_MAX = 400

    # Density Histogram Bins (Global Structure Branch)
    # Ranges:
    # 1. Emphysema: < -950
    # 2. Healthy: -950 to -700
    # 3. Fibrosis/Ground Glass: -700 to -400
    # 4. Consolidation: > -400
    # Edges defined to capture these specific ranges
    DENSITY_BINS = [-2000, -950, -700, -400, 2000]
    DENSITY_BIN_NAMES = ["Emphysema", "Healthy", "Fibrosis", "Consolidation"]

    # Stratified-Variance Sampling
    NUM_ZONES = 3  # Apex, Mid, Base
    SLICES_PER_ZONE = 1  # Select single highest-variance slice per zone

    # ==========================================
    # Feature Engineering
    # ==========================================
    # Dimensionality reduction for texture features
    N_PCA_COMPONENTS = 30

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Quantile Regression
    QUANTILE_ALPHA = 0.5  # Median prediction

    # Training Loop
    NUM_EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3

    # Debugging / Development
    # Set to None to use full dataset, or an integer (e.g., 50) for quick testing
    DEBUG_DATA_SIZE = None

    # ==========================================
    # Metric Constraints
    # ==========================================
    MIN_CONFIDENCE = 70
    MAX_ERROR = 1000

    @staticmethod
    def setup():
        """Ensures necessary writeable directories exist."""
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    @staticmethod
    def get_transforms():
        """
        Returns standard normalization constants for ImageNet pre-trained models.
        Useful if using torchvision transforms.
        """
        return {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
