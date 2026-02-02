import os
import torch
from pathlib import Path


class Config:
    """
    Configuration class for the Hierarchical Dilated Network with Pyramid Context Aggregation (HDN-PCA).
    Centralizes all paths, hyperparameters, and model settings.
    """

    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = Path("./input")
    TRAIN_DIR = INPUT_DIR / "train"
    TEST_DIR = INPUT_DIR / "test"
    METADATA_DIR = Path("./metadata")

    # Working directory for caching processed data and saving models
    WORKING_DIR = Path("./working")

    # Specific cache directory for this idea iteration (Idea 4)
    # This is where cached .npy files and the best model will be stored
    CACHE_DIR = WORKING_DIR / "idea_4"

    # Output path for the submission file
    SUBMISSION_PATH = Path("submission.csv")

    # Path to save/load the best trained model weights
    BEST_MODEL_PATH = CACHE_DIR / "best_model.pth"

    # =========================================================================
    # Data Parameters
    # =========================================================================
    # Z-dimension (depth) of the input volumes
    Z_DIM = 65

    # Patch size for training and validation
    # Using 256x256 as per "Idea" to capture sufficient context
    PATCH_SIZE = 256

    # Stride for sliding window inference (50% overlap recommended)
    INFERENCE_STRIDE = PATCH_SIZE // 2

    # Normalization Statistics (Derived from EDA)
    # Used for Z-score normalization of input volumes
    PIXEL_MEAN = 99.9693
    PIXEL_STD = 12.5444

    # =========================================================================
    # Model Architecture Parameters (HDN-PCA)
    # =========================================================================
    # Dimension of the learnable 2.5D projection (compressing Z=65 -> PROJECTION_DIM)
    PROJECTION_DIM = 32

    # Number of channels in the hierarchical backbone
    BACKBONE_CHANNELS = 64

    # Dilation rates for the sequential residual blocks
    # Hierarchical expansion of receptive field
    BACKBONE_DILATIONS = [1, 2, 4, 8, 16]

    # Dilation rates for the Parallel Atrous Spatial Pyramid Pooling (ASPP) head
    ASPP_RATES = [1, 6, 12, 18]

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42

    # Batch size adjusted for 256x256x65 inputs on available GPU memory
    BATCH_SIZE = 8

    # Learning rate for Adam optimizer
    LEARNING_RATE = 1e-3

    # Maximum number of training epochs
    EPOCHS = 20

    # Early stopping patience (stop if validation F0.5 doesn't improve for N epochs)
    EARLY_STOPPING_PATIENCE = 5

    # Threshold for binarizing predictions (can be tuned during validation)
    INITIAL_THRESHOLD = 0.5

    # =========================================================================
    # Compute & Environment
    # =========================================================================
    NUM_WORKERS = 2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # Debugging / Development
    # =========================================================================
    # If True, runs on a small subset of data for quick pipeline verification
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100

    @classmethod
    def setup(cls):
        """
        Ensures that the necessary working directories exist.
        Should be called at the start of the pipeline.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)


# Automatically create directories when config is imported
Config.setup()
