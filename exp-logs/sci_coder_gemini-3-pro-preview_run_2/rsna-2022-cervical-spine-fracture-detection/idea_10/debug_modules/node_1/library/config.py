import os
import torch


class Config:
    """
    Configuration for the Calibrated 2.5D Dual-Attention Network with Anatomical Injection.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    EXP_NAME = "idea_10"

    # Working directory for outputs and cache
    WORKING_DIR = os.path.join("./working", EXP_NAME)
    os.makedirs(WORKING_DIR, exist_ok=True)

    # =========================================================================
    # Data Paths
    # =========================================================================
    INPUT_ROOT = "./input"

    # Image Directories
    TRAIN_IMAGES_DIR = os.path.join(INPUT_ROOT, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_ROOT, "test_images")
    SEGMENTATION_DIR = os.path.join(INPUT_ROOT, "segmentations")

    # Metadata Files (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Annotations
    BOUNDING_BOX_PATH = os.path.join(INPUT_ROOT, "train_bounding_boxes.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_ROOT, "sample_submission.csv")

    # =========================================================================
    # Model Architecture
    # =========================================================================
    BACKBONE = "efficientnet_b4"
    # Input channels = 3 because we use 2.5D stacking (slices z-1, z, z+1)
    IN_CHANNELS = 3

    # LSTM Settings
    LSTM_HIDDEN_SIZE = 256
    LSTM_LAYERS = 2
    BIDIRECTIONAL = True

    # Dimensions
    FEATURE_DIM = 1792  # EfficientNet-B4 final feature dimension (approx)
    EMBEDDING_DIM = 512  # Dimension after spatial attention projection

    # Outputs
    NUM_TARGETS = 8  # C1-C7 + Patient Overall
    ANATOMY_CLASSES = 8  # Background (0) + C1-C7 (1-7)

    # =========================================================================
    # Input Processing
    # =========================================================================
    IMAGE_SIZE = 512

    # Sequence Settings (High-Density Sampling)
    SEQ_LENGTH = 96  # Number of slices per study fed into the LSTM
    STRIDE = 1  # Stride for sampling slices

    # Normalization
    PIXEL_MEAN = 0.456  # Approximate mean for bone window
    PIXEL_STD = 0.224  # Approximate std for bone window

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    EPOCHS = 10

    # Batch size is small due to large sequence length (96) and 3D nature
    BATCH_SIZE = 2

    # Gradient Accumulation to simulate larger batch size (e.g., effective 16)
    ACCUMULATION_STEPS = 8

    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    MAX_GRAD_NORM = 5.0

    EARLY_STOPPING_PATIENCE = 3

    # =========================================================================
    # Loss Function Weights
    # =========================================================================
    # L_total = L_study + lambda1*L_slice + lambda2*L_spatial + lambda3*L_anatomy

    # Auxiliary Loss Weights
    LAMBDA_FRACTURE = 1.0  # Slice-level fracture detection (BCE)
    LAMBDA_SPATIAL = 1.0  # Spatial attention map guidance (Dice/BCE)
    LAMBDA_ANATOMY = 0.5  # Anatomical level classification (CrossEntropy)

    # Study Level Loss Settings
    # Crucially, we use NO positive weight to ensure probabilistic calibration
    # as per Lesson 00042.
    POS_WEIGHT = 1.0

    # =========================================================================
    # Compute & Environment
    # =========================================================================
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Inference
    TTA = False  # Test Time Augmentation
