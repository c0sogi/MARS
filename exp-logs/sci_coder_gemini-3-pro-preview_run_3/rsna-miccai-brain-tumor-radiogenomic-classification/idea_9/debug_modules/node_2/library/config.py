import os
import torch


class Config:
    """
    Central configuration for the Stabilized High-Density 2.5D Network pipeline.
    Defines paths, data dimensions, and training hyperparameters.
    """

    # ==========================================
    # Path Configurations
    # ==========================================
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")
    METADATA_DIR = "./metadata"

    # Directory for caching processed arrays and saving model checkpoints
    # Corresponds to the current idea iteration
    WORKING_DIR = "./working/idea_9"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # ==========================================
    # Data & Model Dimensions
    # ==========================================
    IMAGE_SIZE = 256
    NUM_SLICES = 32
    NUM_MODALITIES = 4

    # The model accepts a stacked volume of (Slices x Modalities)
    # 32 slices * 4 modalities = 128 input channels
    IN_CHANNELS = NUM_SLICES * NUM_MODALITIES

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 15
    WEIGHT_DECAY = (
        0.0  # Regularization is handled by the stem architecture, not weight decay
    )
    SEED = 42

    # ==========================================
    # System & Compute
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Number of subprocesses for data loading
