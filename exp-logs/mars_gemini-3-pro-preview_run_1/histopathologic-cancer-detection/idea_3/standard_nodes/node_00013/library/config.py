import os
import torch


class Config:
    # -------------------------------------------------------------------------
    # General Configuration
    # -------------------------------------------------------------------------
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Using 4 workers is generally safe and efficient for 12 vCPUs
    NUM_WORKERS = 4

    # -------------------------------------------------------------------------
    # Directory and File Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Output directories - Ensure they exist
    WORKING_DIR = "./working/idea_3"
    SUBMISSION_DIR = "./submission"

    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Model and Submission Output
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Preprocessing & Augmentation
    # -------------------------------------------------------------------------
    # Hard Attention via ROI Cropping: Focus on center 48x48 pixels
    CROP_SIZE = 48

    # Input resolution for the model (matches crop size)
    IMAGE_SIZE = 48

    # Normalization (ImageNet Statistics)
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]

    # Augmentation Parameters
    # Geometric transformations
    PROB_HFLIP = 0.5
    PROB_VFLIP = 0.5
    PROB_ROTATE = 0.5  # For random 90-degree rotations

    # Color Augmentation (Restricted)
    # Hue and Saturation are disabled to preserve H&E stain signatures
    COLOR_JITTER_BRIGHTNESS = 0.1
    COLOR_JITTER_CONTRAST = 0.1
    COLOR_JITTER_SATURATION = 0.0
    COLOR_JITTER_HUE = 0.0

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    MODEL_NAME = "convnext_tiny"
    PRETRAINED = True
    NUM_CLASSES = 1

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 32
    EPOCHS = 20

    # Optimizer (AdamW)
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.01

    # Scheduler (Cosine Annealing)
    ETA_MIN = 1e-6

    # Early Stopping
    # Relaxed patience to allow model to stabilize after volatile phases
    PATIENCE = 6

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------
    # Test Time Augmentation (TTA) enabled
    TTA_ENABLED = True
