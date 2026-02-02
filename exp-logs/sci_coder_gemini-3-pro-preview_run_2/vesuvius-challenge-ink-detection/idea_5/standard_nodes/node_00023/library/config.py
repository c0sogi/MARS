import os
import torch


class Config:
    # ==============================
    # File Paths
    # ==============================
    # Root directory for input data (Read-Only)
    INPUT_DIR = "./input"

    # Directory containing pre-generated metadata CSVs
    METADATA_DIR = "./metadata"

    # Working directory for caching processed data and saving models
    # Specific to 'idea_5' to avoid overwriting other experiments
    WORKING_DIR = "./working/idea_5"

    # Path for the final submission file (root directory as per competition format)
    SUBMISSION_PATH = "./submission/submission.csv"

    # ==============================
    # Data Configuration
    # ==============================
    TILE_SIZE = 512

    # Stratified Depth Projection (2.5D) Strategy
    # The central Z-volume (slices 22-42) is split into 3 sub-volumes of 7 slices each.
    # Each tuple represents (start_index, end_index) for slicing (end is exclusive).
    # These will be processed into MIPs and stacked to form a 3-channel input.
    # Channel 0: Slices 22-28
    # Channel 1: Slices 29-35
    # Channel 2: Slices 36-42
    Z_RANGES = [(22, 29), (29, 36), (36, 43)]

    # Input channels corresponds to the number of stratified layers
    IN_CHANNELS = len(Z_RANGES)

    # ==============================
    # Model Configuration
    # ==============================
    # Architecture: SegFormer with Mix Transformer B1 (MiT-B1) encoder
    MODEL_ARCH = "mit_b1"
    NUM_CLASSES = 1

    # ==============================
    # Training Hyperparameters
    # ==============================
    SEED = 42
    BATCH_SIZE = 16
    NUM_EPOCHS = 15

    # Learning Rate for Transformer-based encoder
    # (Transformers typically require lower LR than CNNs)
    LEARNING_RATE = 6e-5

    # Optimizer settings (AdamW)
    WEIGHT_DECAY = 0.01

    # Scheduler settings (ReduceLROnPlateau)
    PATIENCE = 3
    FACTOR = 0.5

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # ==============================
    # Inference Configuration
    # ==============================
    # Threshold for binarizing probability maps
    THRESHOLD = 0.5

    # Enable Test Time Augmentation (Flip/Rotate)
    USE_TTA = True

    @classmethod
    def setup(cls):
        """
        Ensures that the working directory exists.
        This is called automatically when the config is imported.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)


# Automatically execute setup on import
Config.setup()
