import os
import torch


class Config:
    """
    Configuration for Salt Segmentation Task.
    Implements the strategy: U-Net++ with ResNeXt-50, Input Channel Multiplexing,
    and Two-Stage Loss Curriculum (BCE+Dice -> Lovasz).
    """

    # =========================================================================
    # Directories & Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_6"

    # Sub-directories
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = "./submission"

    # Input Files (Metadata)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
    DEPTHS_CSV_PATH = os.path.join(INPUT_DIR, "depths.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Files
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Files (Numpy format for fast loading)
    CACHE_TRAIN_IMAGES = os.path.join(CACHE_DIR, "train_images.npy")
    CACHE_TRAIN_MASKS = os.path.join(CACHE_DIR, "train_masks.npy")
    CACHE_TRAIN_DEPTHS = os.path.join(CACHE_DIR, "train_depths.npy")

    CACHE_VAL_IMAGES = os.path.join(CACHE_DIR, "val_images.npy")
    CACHE_VAL_MASKS = os.path.join(CACHE_DIR, "val_masks.npy")
    CACHE_VAL_DEPTHS = os.path.join(CACHE_DIR, "val_depths.npy")

    CACHE_TEST_IMAGES = os.path.join(CACHE_DIR, "test_images.npy")
    CACHE_TEST_DEPTHS = os.path.join(CACHE_DIR, "test_depths.npy")
    CACHE_TEST_IDS = os.path.join(CACHE_DIR, "test_ids.npy")

    # =========================================================================
    # Data Parameters
    # =========================================================================
    ORIG_SIZE = 101
    IMG_SIZE = 128  # Padded size (Reflection padding 101 -> 128)
    CHANNELS = 3  # Input: [Seismic, Seismic, Depth]
    NUM_WORKERS = 4  # Number of data loading workers

    # =========================================================================
    # Model Parameters
    # =========================================================================
    ENCODER = "se_resnext50_32x4d"
    ENCODER_WEIGHTS = "imagenet"
    DECODER_CHANNELS = (256, 128, 64, 32, 16)

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4

    # Loss Curriculum
    # Epoch < LOVASZ_SWITCH_EPOCH: BCE + Dice Loss (Warmup)
    # Epoch >= LOVASZ_SWITCH_EPOCH: Lovasz-Hinge Loss (Fine-tuning)
    LOVASZ_SWITCH_EPOCH = 15

    # Scheduler (ReduceLROnPlateau)
    SCHEDULER_PATIENCE = 5
    SCHEDULER_FACTOR = 0.5
    MIN_LR = 1e-6

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 12

    # =========================================================================
    # Inference Parameters
    # =========================================================================
    TTA_FLIP = True  # Test Time Augmentation: Horizontal Flip
    IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    @classmethod
    def setup(cls):
        """
        Initialize the working environment.
        Creates necessary directories for checkpoints, caching, and submissions.
        """
        dirs = [cls.WORKING_DIR, cls.CHECKPOINT_DIR, cls.CACHE_DIR, cls.SUBMISSION_DIR]
        for d in dirs:
            os.makedirs(d, exist_ok=True)


# Execute setup on import to ensure environment is ready
Config.setup()
