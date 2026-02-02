import os
import torch
import numpy as np
import random


class Config:
    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_10"
    SUBMISSION_DIR = "./submission"

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    LOG_PATH = os.path.join(WORKING_DIR, "training_log.csv")

    # =========================================================================
    # Data Configuration
    # =========================================================================
    IMAGE_SIZE = 1024  # Letterbox resize target dimension
    NUM_CHANNELS = 3  # Input channels (converted from grayscale)

    # Labels
    NUM_CLASSES = 1  # Object detection class: 'opacity'
    CLASS_LABELS = ["opacity"]

    # Study Level Labels (Order matches train_study_level.csv columns)
    STUDY_LABELS = [
        "Negative for Pneumonia",
        "Typical Appearance",
        "Indeterminate Appearance",
        "Atypical Appearance",
    ]
    NUM_STUDY_CLASSES = 4

    # Data Loading
    NUM_WORKERS = 4  # Number of DataLoader workers
    PIN_MEMORY = True

    # =========================================================================
    # Model Configuration (Co-DETR + Swin-L)
    # =========================================================================
    # Backbone
    BACKBONE_NAME = "swin_large_patch4_window12_384"  # timm model name
    BACKBONE_PRETRAINED = True

    # Transformer
    HIDDEN_DIM = 256
    NHEADS = 8
    NUM_ENCODER_LAYERS = 6
    NUM_DECODER_LAYERS = 6
    DIM_FEEDFORWARD = 1024
    DROPOUT = 0.1

    # Queries
    NUM_QUERIES = 100  # Number of object queries

    # =========================================================================
    # Training Configuration
    # =========================================================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Hyperparameters
    BATCH_SIZE = 4  # Adjusted for A100 40GB with Swin-L
    EPOCHS = 10  # Total training epochs

    # Optimization
    LR = 1e-4  # Base learning rate
    BACKBONE_LR = 1e-5  # Lower LR for backbone
    WEIGHT_DECAY = 1e-4
    CLIP_MAX_NORM = 0.1  # Gradient clipping
    LR_DROP = 7  # Epoch to decay LR

    # Loss Weights (Co-DETR Multi-Task)
    # Matcher costs
    COST_CLASS = 2.0
    COST_BBOX = 5.0
    COST_GIOU = 2.0

    # Loss components
    LAMBDA_DETR = 1.0  # Main decoder loss
    LAMBDA_AUX_ATSS = 1.0  # Auxiliary ATSS head loss
    LAMBDA_AUX_RCNN = 1.0  # Auxiliary Faster R-CNN head loss
    LAMBDA_STUDY = 1.0  # Study-level classification loss

    # =========================================================================
    # Inference & Post-processing
    # =========================================================================
    CONF_THRESHOLD = 0.001  # Confidence threshold (low for mAP calculation)
    IOU_THRESHOLD = 0.5  # IoU threshold for evaluation

    # =========================================================================
    # Debugging
    # =========================================================================
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use in debug mode

    @classmethod
    def setup(cls, debug=False):
        """
        Initializes the environment:
        1. Creates necessary directories.
        2. Sets random seeds for reproducibility.
        3. Updates config for debug mode if requested.
        """
        # Create directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set seeds
        cls.seed_everything(cls.SEED)

        # Handle Debug Mode
        if debug:
            cls.DEBUG = True
            cls.EPOCHS = 2
            print(
                f"Config: Debug mode enabled. Training for {cls.EPOCHS} epochs on {cls.DEBUG_SAMPLE_SIZE} samples."
            )

    @staticmethod
    def seed_everything(seed):
        """Sets seeds for all random number generators."""
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @classmethod
    def get_transforms_config(cls):
        """Returns dictionary of transform settings for Albumentations."""
        return {
            "image_size": cls.IMAGE_SIZE,
            "scale_limit": (0.1, 2.0),  # LSJ Scale range
            "p_flip": 0.5,
        }
