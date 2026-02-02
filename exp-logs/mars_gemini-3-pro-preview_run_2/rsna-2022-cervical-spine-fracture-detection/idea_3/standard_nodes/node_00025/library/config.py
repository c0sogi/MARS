import os
import torch


class Config:
    """
    Central configuration for the Cervical Spine Fracture Detection task.
    Defines hyperparameters, file paths, and model settings for the
    2.5D Multi-Head Attention Network.
    """

    # =========================================================================
    # Directories & Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_3"
    OUTPUT_DIR = "./submission"

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Image Directories
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    # Output Files
    SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")
    MODEL_CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Cache Files (Parquet)
    TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_cache.parquet")
    VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_cache.parquet")
    TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_cache.parquet")

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    # Backbone CNN
    BACKBONE_NAME = "efficientnet_b3"
    IMAGE_SIZE = 384
    IN_CHANNELS = 3  # 2.5D Input: Slices [z-1, z, z+1]

    # Sequence Modeling (LSTM)
    SEQ_LEN = 48  # Number of slices sampled per study
    LSTM_HIDDEN_SIZE = 256
    LSTM_LAYERS = 2
    DROPOUT = 0.2

    # Heads
    NUM_CLASSES = 8  # C1-C7 (7 classes) + Patient Overall (1 class)

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    EPOCHS = 10

    # Batch Size & Optimization
    # Effective Batch Size = BATCH_SIZE * GRAD_ACCUM_STEPS
    # Using small batch size due to large image size (384) and sequence length (48)
    BATCH_SIZE = 1
    GRAD_ACCUM_STEPS = 16

    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    MAX_GRAD_NORM = 5.0

    # Loss Function
    # Positive weight > 1 to prioritize fracture detection (sensitivity)
    POS_WEIGHT = 2.0

    # =========================================================================
    # Compute & Debugging
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # Debugging flags to speed up development
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 20  # Number of samples to use if DEBUG is True

    @classmethod
    def setup(cls):
        """
        Ensures that the necessary working and output directories exist.
        Should be called at the start of any script using this config.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)

        # Print configuration summary
        print(f"Configuration Loaded:")
        print(f"  Device: {cls.DEVICE}")
        print(f"  Backbone: {cls.BACKBONE_NAME}")
        print(f"  Image Size: {cls.IMAGE_SIZE}x{cls.IMAGE_SIZE}")
        print(f"  Sequence Length: {cls.SEQ_LEN}")
        print(f"  Effective Batch Size: {cls.BATCH_SIZE * cls.GRAD_ACCUM_STEPS}")
        print(f"  Working Dir: {cls.WORKING_DIR}")
