import os
import torch


class Config:
    """
    Central configuration for the Cervical Spine Fracture Detection pipeline.
    Implements the 'Stabilized 2.5D ConvNeXt Multi-Task MIL Network' strategy.
    """

    # --- General ---
    DEBUG = False
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Use available vCPUs
    NUM_WORKERS = 12

    # --- File Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Metadata Files (Generated previously)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Image Directories
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    # Sample Submission
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # --- Caching ---
    # Directory for storing preprocessed 2.5D stacks (uint8)
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_15")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # --- DICOM Preprocessing ---
    # Standard Bone Window
    WINDOW_LEVEL = 400
    WINDOW_WIDTH = 1800

    # Input Dimensions
    IMAGE_SIZE = 224
    # 2.5D Stacking: Input channels = 3 (slices z-1, z, z+1)
    IN_CHANNELS = 3
    # Volume Depth: Uniformly sample 64 slices per exam
    NUM_SLICES = 64

    # --- Model Architecture ---
    # Backbone: ConvNeXt-Tiny (uses LayerNorm for batch stability)
    MODEL_NAME = "convnext_tiny"
    PRETRAINED = True
    # Targets: C1-C7 (7) + Patient Overall (1) = 8
    NUM_CLASSES = 8

    # --- Training Hyperparameters ---
    # Batch Size 8 is critical for stability with this architecture/VRAM
    BATCH_SIZE = 8
    EPOCHS = 10

    # Optimizer settings
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-2

    # Scheduler: Decoupled Cosine Annealing
    # T_max is set to 1.5x the number of epochs to prevent premature decay
    T_MAX_MULT = 1.5

    # --- Debugging / Development ---
    # If set to an integer, limits the number of samples loaded for quick testing
    DEBUG_DATA_SIZE = None

    # --- Submission ---
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
