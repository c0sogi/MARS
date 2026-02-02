import os
import torch


class Config:
    """
    Global configuration for the Salt Segmentation task.
    Includes paths, hyperparameters, and model settings based on ResNet34-LinkNet with ASPP.
    """

    # ==========================
    # General Settings
    # ==========================
    SEED = 42
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 50  # Number of samples to use in debug mode

    # ==========================
    # Paths
    # ==========================
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"

    # Metadata files (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Raw Data (for reference, though metadata contains paths)
    DEPTHS_CSV = os.path.join(INPUT_ROOT, "depths.csv")

    # Output Paths
    WORKING_DIR = "./working/idea_4"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================
    # Data Parameters
    # ==========================
    ORIG_SIZE = 101
    INPUT_SIZE = 128  # Pad 101x101 to 128x128 for UNet-like architectures (powers of 2)
    IN_CHANNELS = 1  # Grayscale input
    NUM_CLASSES = 1  # Binary segmentation (Salt vs Sediment)

    # ==========================
    # Model Architecture
    # ==========================
    BACKBONE = "resnet34"
    ENCODER_WEIGHTS = "imagenet"
    DEPTH_EMBEDDING_DIM = 32  # Dimension for depth injection vector

    # ==========================
    # Training Hyperparameters
    # ==========================
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 12

    # Scheduler (Cosine Annealing)
    T_MAX = 50
    ETA_MIN = 1e-6

    # Loss Weights
    BCE_WEIGHT = 0.5
    LOVASZ_WEIGHT = 0.5

    # ==========================
    # Augmentation
    # ==========================
    AUG_PROB = 0.5
    AUG_ELASTIC_ALPHA = 120.0
    AUG_ELASTIC_SIGMA = 6.0

    # ==========================
    # Inference
    # ==========================
    TTA_FLIP = True  # Test Time Augmentation: Horizontal Flip
    THRESHOLD_START = 0.5  # Starting point for threshold optimization

    # ==========================
    # Hardware
    # ==========================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    @classmethod
    def create_dirs(cls):
        """Creates necessary directories for outputs and cache."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


def get_config(debug=False, epochs=None, batch_size=None):
    """
    Retrieves the configuration class, optionally overriding parameters.

    Args:
        debug (bool): If True, sets DEBUG flag and reduces epochs/batch_size for quick testing.
        epochs (int, optional): Override the number of training epochs.
        batch_size (int, optional): Override the batch size.

    Returns:
        Type[Config]: The configured Config class.
    """
    # Create a copy or just modify the class (since we don't run concurrent configs)
    # Modifying the class directly is sufficient for this script structure.
    cfg = Config

    if debug:
        cfg.DEBUG = True
        cfg.EPOCHS = 2
        cfg.BATCH_SIZE = 8
        print(f"DEBUG MODE ENABLED: Epochs={cfg.EPOCHS}, Batch={cfg.BATCH_SIZE}")

    if epochs is not None:
        cfg.EPOCHS = epochs

    if batch_size is not None:
        cfg.BATCH_SIZE = batch_size

    # Ensure directories exist
    cfg.create_dirs()

    return cfg
