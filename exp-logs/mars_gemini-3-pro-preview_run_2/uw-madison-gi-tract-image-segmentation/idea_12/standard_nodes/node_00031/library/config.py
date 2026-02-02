import os
import torch


class Config:
    """
    Configuration for Idea 12: 2.5D Coarse-to-Fine Cascade Network.
    """

    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    METADATA_DIR = "./metadata"
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Working directory for Idea 12 (Cache and Checkpoints)
    WORKING_DIR = "./working/idea_12"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Submission directory
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Model Checkpoints
    COARSE_MODEL_PATH = os.path.join(WORKING_DIR, "best_coarse_model.pth")
    FINE_MODEL_PATH = os.path.join(WORKING_DIR, "best_fine_model.pth")

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Number of data loading workers (12 vCPUs available)
    NUM_WORKERS = 4
    # Set to True to run on a small subset for debugging
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 200

    # =========================================================================
    # Data Configuration
    # =========================================================================
    # 2.5D Stacking: Input channels = 3 (Slice i-1, Slice i, Slice i+1)
    IN_CHANNELS = 3
    NUM_CLASSES = 3  # large_bowel, small_bowel, stomach
    CLASS_LABELS = ["large_bowel", "small_bowel", "stomach"]

    # =========================================================================
    # Stage 1: Coarse Model (Global Localization)
    # =========================================================================
    # Architecture: 2.5D Ghost U-Net
    COARSE_BACKBONE = "ghostnet"

    # Input: Downsampled full slice
    COARSE_IMG_SIZE = (256, 256)  # (Height, Width)

    # Training Hyperparameters
    COARSE_BATCH_SIZE = 32
    COARSE_EPOCHS = 15
    COARSE_LR = 1e-3
    COARSE_WD = 1e-5

    # Loss Weights (BCE + Dice)
    COARSE_BCE_WEIGHT = 0.5
    COARSE_DICE_WEIGHT = 0.5

    # =========================================================================
    # Stage 2: Fine Model (Local Refinement)
    # =========================================================================
    # Architecture: 2.5D EfficientNet-B1 U-Net
    FINE_BACKBONE = "efficientnet-b1"

    # Input: Cropped ROI
    FINE_IMG_SIZE = (320, 320)  # (Height, Width)

    # Training Hyperparameters
    FINE_BATCH_SIZE = 16
    FINE_EPOCHS = 20
    FINE_LR = 1e-4
    FINE_WD = 1e-4

    # Loss Params for Tversky Loss
    # Alpha (FP) < Beta (FN) to prioritize Recall
    TVERSKY_ALPHA = 0.3
    TVERSKY_BETA = 0.7
    TVERSKY_SMOOTH = 1.0

    # =========================================================================
    # ROI & Inference Logic
    # =========================================================================
    # Margin to add around the coarse bounding box (e.g., 0.15 = 15% expansion)
    ROI_MARGIN_RATIO = 0.15

    # Threshold to convert probability maps to binary masks
    MASK_THRESHOLD = 0.5

    # Post-processing: Minimum pixel count to keep a mask component
    MIN_PIXEL_COUNT = 10
