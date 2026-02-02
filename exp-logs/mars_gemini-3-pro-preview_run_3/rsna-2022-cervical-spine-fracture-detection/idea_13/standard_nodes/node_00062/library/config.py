import os
import torch


class Config:
    """
    Configuration for the Dual-Resolution ConvNeXt Multi-Task MIL Network.
    Centralizes all hyperparameters, file paths, and model settings.
    """

    # =========================================================================
    # Path Configuration
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Cache directory for preprocessed 2.5D stacks (Idea 13)
    # Stores .npy files to maximize I/O throughput
    CACHE_DIR = "./working/idea_13"

    # Output directory for predictions
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Preprocessing & Augmentation
    # =========================================================================
    # Input Channels: 3 (2.5D Stacking: z-1, z, z+1)
    IN_CHANS = 3

    # Spatial Dimensions
    # Global Stream: Resized to 256x256
    # Local Stream: Center Cropped to 256x256 (Preserves high freq details)
    IMAGE_SIZE = 256

    # Volumetric Sampling
    # Uniformly sample 64 slices per exam to form the input bag
    NUM_SLICES = 64

    # Hounsfield Unit (HU) Windowing - Standard Bone Window
    WINDOW_LEVEL = 400
    WINDOW_WIDTH = 1800

    # =========================================================================
    # Model Architecture
    # =========================================================================
    # Backbone: ConvNeXt-Tiny (LayerNorm-native for stability with small batches)
    BACKBONE = "convnext_tiny"
    PRETRAINED = True

    # Output Heads: 7 Vertebrae (C1-C7) + 1 Patient Overall
    NUM_CLASSES = 8

    # Feature Dimension for ConvNeXt Tiny
    HIDDEN_DIM = 768

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    # Random Seed for Reproducibility
    SEED = 42

    # Batch Size
    # Set to 2 to accommodate the memory footprint of Dual-Stream inputs
    BATCH_SIZE = 2

    # Optimization
    EPOCHS = 10
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-2
    MAX_GRAD_NORM = 1000.0

    # Learning Rate Scheduler (Decoupled Cosine Annealing)
    # T_max is set to 1.5x epochs to prevent premature convergence
    T_MAX_MULT = 1.5
    MIN_LR = 1e-6

    # Regularization
    # No dropout in heads to preserve signal in MIL setting
    DROP_RATE = 0.0
    DROP_PATH_RATE = 0.0

    # =========================================================================
    # Compute & Environment
    # =========================================================================
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # Debugging & Development
    # =========================================================================
    # Toggle DEBUG to True to run on a small subset of data
    DEBUG = False

    # If not None, limits the number of samples used for training/validation
    # Useful for quick pipeline verification
    N_SAMPLES = None

    @classmethod
    def create_directories(cls):
        """
        Ensures that the necessary working directories exist.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Automatically create directories when config is imported
Config.create_directories()
