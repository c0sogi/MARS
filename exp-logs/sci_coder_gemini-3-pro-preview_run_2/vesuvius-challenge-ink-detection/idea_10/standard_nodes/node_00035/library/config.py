import os
import torch


class Config:
    """
    Configuration for Idea 10: Z-Scanning SegFormer (MiT-B4).
    Centralizes all parameters for data processing, model architecture,
    training, and inference protocols.
    """

    # =========================================================================
    # File System Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Specific cache directory for this idea
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_10")

    # Metadata files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VALIDATION_METADATA_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Final submission file location
    SUBMISSION_PATH = "./submission.csv"

    # =========================================================================
    # Data Preprocessing & Volumetric Slicing
    # =========================================================================
    TILE_SIZE = 512

    # Standard Narrow Context Base Index
    # Channel 1 starts at 20, Ch2 at 26, Ch3 at 32
    Z_START = 20

    # Projection Parameters
    Z_DIM = 12  # Number of slices per channel (Depth)
    Z_CHANNEL_STRIDE = 6  # Overlap stride between channels

    # Input definition
    IN_CHANNELS = 3

    # =========================================================================
    # Augmentation
    # =========================================================================
    # Volumetric Z-Jitter: Shift start index by random value in [-5, 5]
    Z_JITTER = 5

    # =========================================================================
    # Model Architecture
    # =========================================================================
    ENCODER_NAME = "mit_b4"
    ENCODER_WEIGHTS = "imagenet"
    DECODER_CHANNELS = [256, 128, 64, 32]
    CLASSES = 1
    ACTIVATION = "sigmoid"

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    # Batch size 8 is conservative for MiT-B4 @ 512x512 on 40GB VRAM
    BATCH_SIZE = 8
    NUM_WORKERS = 4
    EPOCHS = 15
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-2

    # Validation Gating: Only save/submit if score > 0.5976
    VALIDATION_THRESHOLD = 0.5976

    # =========================================================================
    # Inference Strategy: Multi-Pass Z-Scanning
    # =========================================================================
    # Offsets relative to Z_START to scan for wandering ink
    INFERENCE_Z_OFFSETS = [-4, 0, 4]

    # =========================================================================
    # Compute
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Initialize the working environment.
        Creates the cache directory if it does not exist.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)

    @classmethod
    def get_channel_indices(cls, base_start_index):
        """
        Generates the list of (start, end) slice tuples for the 3 input channels
        based on a provided base start index.

        This handles:
        1. Standard training (base_start_index = Z_START)
        2. Training with Jitter (base_start_index = Z_START + random_offset)
        3. Inference Scanning (base_start_index = Z_START + scan_offset)

        Args:
            base_start_index (int): The starting slice index for the first channel.

        Returns:
            list of tuple: [(start1, end1), (start2, end2), (start3, end3)]
        """
        slices = []
        current_start = base_start_index

        for _ in range(cls.IN_CHANNELS):
            slices.append((current_start, current_start + cls.Z_DIM))
            current_start += cls.Z_CHANNEL_STRIDE

        return slices
