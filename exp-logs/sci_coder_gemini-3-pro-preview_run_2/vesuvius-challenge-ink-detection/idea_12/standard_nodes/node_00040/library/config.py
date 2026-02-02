import os
import torch


class Config:
    """
    Configuration for Stabilized High-Fidelity SegFormer (MiT-B3) with Decoupled Z-Scanning.
    """

    # ==============================
    # Paths & Directories
    # ==============================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Working directory for caching processed tensors and model checkpoints
    WORKING_DIR = "./working/idea_12"
    # Output path for the final submission file
    SUBMISSION_PATH = "submission/submission.csv"

    # ==============================
    # Data Generation & Z-Slicing
    # ==============================
    TILE_SIZE = 512

    # Training Z-Slice Configuration
    # We use a fixed narrow context centered around slice 32 (range 20-44).
    # This avoids the "wandering ink" noise found in wider contexts during training.
    TRAIN_Z_START = 20

    # Relative offsets for the 3-channel overlapping slabs.
    # Each tuple represents (start_offset, end_offset) relative to the base Z_START.
    # Channel 1: Base + 0 to Base + 12
    # Channel 2: Base + 6 to Base + 18
    # Channel 3: Base + 12 to Base + 24
    Z_OFFSETS = [(0, 12), (6, 18), (12, 24)]

    # Inference Z-Scanning Configuration
    # To handle ink that wanders in depth, we scan the test fragments at multiple base depths.
    # The model predictions from these scans will be fused (Max Projection).
    INFERENCE_Z_STARTS = [18, 20, 22]

    # ==============================
    # Model Architecture
    # ==============================
    # Using SegFormer MiT-B3 for higher capacity and texture resolution
    MODEL_BACKBONE = "nvidia/mit-b3"
    PRETRAINED = True
    IN_CHANNELS = 3
    NUM_CLASSES = 1

    # ==============================
    # Optimization & Training
    # ==============================
    SEED = 42

    # Conservative Learning Rate to ensure stability with the larger backbone
    LEARNING_RATE = 6e-5
    WEIGHT_DECAY = 1e-2

    # Training Loop Parameters
    BATCH_SIZE = 8  # Adjusted for A100 GPU memory with MiT-B3
    NUM_EPOCHS = 15

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 4
    EARLY_STOPPING_MIN_DELTA = 0.001

    # Learning Rate Scheduler (ReduceLROnPlateau)
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 2

    # Validation Logic
    # Minimum F0.5 score required to generate a submission file (Baseline check)
    BASELINE_SCORE_THRESHOLD = 0.598

    # ==============================
    # Compute & Infrastructure
    # ==============================
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Initializes the environment by ensuring necessary directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)


# Initialize setup on import
Config.setup()
