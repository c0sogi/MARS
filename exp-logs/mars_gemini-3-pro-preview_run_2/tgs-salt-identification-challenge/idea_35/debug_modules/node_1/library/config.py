import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration for Salt Segmentation Task.
    Implements the Marginalized-Scan Multi-Task Distillation strategy settings.
    """

    # =========================================================================
    # Paths
    # =========================================================================
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Cache directory for deterministic data processing (Idea 35)
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_35")

    # Submission directory and path
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata File Paths
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")
    DEPTHS_CSV = os.path.join(INPUT_ROOT, "depths.csv")

    # =========================================================================
    # Data & Image Processing
    # =========================================================================
    IMG_ORIG_SIZE = 101
    IMG_TARGET_SIZE = 128  # Padded/Resized to be divisible by 32 for ResNet
    CHANNELS = 1  # Grayscale seismic images

    # Normalization (Standard ImageNet stats)
    NORM_MEAN = (0.485, 0.456, 0.406)
    NORM_STD = (0.229, 0.224, 0.225)

    # =========================================================================
    # Model Architecture
    # =========================================================================
    BACKBONE = "resnet34"
    ENCODER_WEIGHTS = "imagenet"

    # Specialist Teacher: Injects depth into the bottleneck
    TEACHER_USE_DEPTH = True

    # Generalist Student: Image-only input, but uses Aux head for training
    STUDENT_USE_DEPTH = False
    STUDENT_AUX_HEAD = True

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Optimization
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    OPTIMIZER = "AdamW"
    SCHEDULER = "CosineAnnealingLR"

    # Batching
    BATCH_SIZE = 32
    NUM_WORKERS = 4

    # Precision: Strictly FP32 to avoid instability with ranking losses
    USE_AMP = False

    # Training Duration (Overridable)
    EPOCHS_STAGE1 = 50  # Teacher Ensemble Training
    EPOCHS_STAGE3 = 50  # Student Distillation

    # =========================================================================
    # Augmentation (Albumentations)
    # =========================================================================
    # Elastic Transform settings (Crucial for this task)
    AUG_ELASTIC_ALPHA = 120
    AUG_ELASTIC_SIGMA = 6
    AUG_ELASTIC_ALPHA_AFFINE = 3.6  # 120 * 0.03

    AUG_PROB = 0.2  # Probability of applying augmentations

    # =========================================================================
    # Strategy: Marginalized-Scan & Distillation
    # =========================================================================
    N_FOLDS = 5

    # Depth Scan Range: Standard deviations to scan for marginalized inference
    # Used to generate robust pseudo-labels for the test set
    DEPTH_SCAN_RANGE = [-1.5, -0.75, 0.0, 0.75, 1.5]

    # =========================================================================
    # Debug & Runtime Control
    # =========================================================================
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use in debug mode

    def __init__(self, **kwargs):
        """
        Initialize the configuration with optional overrides.

        Args:
            **kwargs: Key-value pairs to override default configuration attributes.
                      Useful for setting DEBUG=True, changing EPOCHS, or BATCH_SIZE.
        """
        # Update attributes with kwargs
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                # Allow setting new attributes if necessary, but typically we override existing
                setattr(self, k, v)

        # Ensure essential directories exist
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        os.makedirs(self.SUBMISSION_DIR, exist_ok=True)

        # Enforce reproducibility immediately
        self.setup_reproducibility(self.SEED)

    @staticmethod
    def setup_reproducibility(seed):
        """
        Sets fixed random seeds for reproducibility across libraries.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Python hash seed
        os.environ["PYTHONHASHSEED"] = str(seed)
