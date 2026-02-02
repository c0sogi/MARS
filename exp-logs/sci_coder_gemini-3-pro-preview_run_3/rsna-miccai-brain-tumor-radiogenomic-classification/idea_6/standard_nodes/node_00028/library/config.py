import os
import torch


class Config:
    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_opt"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # ==========================================
    # Data Configuration
    # ==========================================
    IMG_SIZE = 256

    # Total slices to extract from the volume (10%-90% range)
    NUM_SLICES_TOTAL = 32

    # Slices per stream (Even/Odd split means half of total)
    NUM_SLICES_PER_STREAM = 16

    # Input channels for the network
    # Calculation: 16 slices * 4 modalities = 64 channels
    INPUT_CHANNELS = 64

    # ==========================================
    # Training Configuration
    # ==========================================
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 15
    SEED = 42

    # ==========================================
    # Compute Configuration
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    @classmethod
    def to_dict(cls):
        """Returns configuration as a dictionary."""
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__") and not callable(v)
        }
