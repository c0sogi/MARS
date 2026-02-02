import os
import torch


class Config:
    """
    Configuration for Apple Disease Detection (Idea 4).

    Strategy:
    - Model: ConvNeXt-Small (384x384 resolution)
    - Training: 25 Epochs, BCEWithLogitsLoss, EMA (0.9999), DropPath (0.2)
    - Augmentation: RandomResizedCrop (0.5), Vertical+Horizontal Flip, ColorJitter (0.2)
    - Hardware: A100 GPU Support (AMP enabled implicitly in training loop logic)
    """

    # ==========================================
    # Experiment Identity
    # ==========================================
    EXP_NAME = "idea_4"
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging

    # ==========================================
    # Directories & Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific experiment idea
    WORKING_DIR = os.path.join("./working", EXP_NAME)
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")

    # Submission output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Data Sources
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Ensure necessary writable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Data Configuration
    # ==========================================
    IMG_SIZE = 384
    NUM_CLASSES = 6
    # Alphabetical order matching the metadata generation logic
    CLASSES = [
        "complex",
        "frog_eye_leaf_spot",
        "healthy",
        "powdery_mildew",
        "rust",
        "scab",
    ]

    # Augmentation Hyperparameters
    AUG_SCALE_MIN = 0.5  # Aggressive cropping to capture local disease features
    AUG_SCALE_MAX = 1.0
    AUG_COLOR_JITTER = 0.2  # Brightness and Contrast strength

    # ==========================================
    # Model Configuration
    # ==========================================
    # Using ConvNeXt-Small pretrained on 22k and finetuned on 1k at 384x384
    MODEL_NAME = "convnext_small.fb_in22k_ft_in1k_384"

    # Regularization
    DROP_PATH_RATE = 0.2

    # ==========================================
    # Training Configuration
    # ==========================================
    EPOCHS = 25
    BATCH_SIZE = 32  # Conservative batch size for 384x384 on A100
    LEARNING_RATE = 2e-4  # Standard fine-tuning LR for ConvNeXt
    WEIGHT_DECAY = 1e-2  # Standard AdamW weight decay

    # Exponential Moving Average (EMA)
    USE_EMA = True
    EMA_DECAY = 0.9999

    # ==========================================
    # Hardware & Compute
    # ==========================================
    NUM_WORKERS = 12
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # Inference / Post-Processing
    # ==========================================
    CONF_THRESHOLD = 0.5
    TTA_FLIP = True  # Perform Horizontal and Vertical Flip TTA
