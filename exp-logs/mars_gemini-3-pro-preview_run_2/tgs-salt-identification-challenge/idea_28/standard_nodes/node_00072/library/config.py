import os
import torch


class Config:
    """
    Central configuration for the Salt Segmentation Task.
    Implements the 'Corrected Multi-Task Wide-LinkNet with Gated Noisy Student' strategy.
    """

    # =========================================================================
    # Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    # Specific cache directory for this idea to ensure directory safety
    CACHE_DIR = "./working/idea_28"
    SUBMISSION_DIR = "./submission"

    # Metadata files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    DEPTHS_CSV = os.path.join(INPUT_DIR, "depths.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # =========================================================================
    # Data Parameters
    # =========================================================================
    ORIG_SIZE = 101
    IMG_SIZE = 128  # Padded size (multiple of 32 for ResNet)
    CHANNELS = 1  # We sum RGB weights to make it 1-channel

    # Normalization (Standard ImageNet)
    # Even though we convert to 1-channel, we often use the mean of these or
    # apply them before summing if using a library, but here we define them for reference.
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 50
    NUM_FOLDS = 5

    # Debugging / Development
    DEBUG = False
    DEBUG_SIZE = 100  # Number of samples to use when DEBUG is True

    # =========================================================================
    # Model Configuration
    # =========================================================================
    BACKBONE = "resnet34"
    PRETRAINED = True

    # Multi-Task Loss Weights
    # Loss = Lovasz + BCE + (DEPTH_LOSS_WEIGHT * MSE)
    DEPTH_LOSS_WEIGHT = 0.1

    # Gating for Stage 2 (Noisy Student)
    # Models with validation mAP < MAP_THRESHOLD are discarded from the ensemble
    MAP_THRESHOLD = 0.75

    # =========================================================================
    # Augmentation Parameters
    # =========================================================================
    # Elastic Transform
    ELASTIC_ALPHA = 120
    ELASTIC_SIGMA = 6
    ELASTIC_ALPHA_AFFINE = 120 * 0.03

    # Rigid Transform
    SHIFT_SCALE_ROTATE_P = 0.2

    # =========================================================================
    # Hardware
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    @classmethod
    def setup(cls):
        """
        Creates necessary working directories.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def get_param_str(cls):
        """Returns a string representation of key params for logging."""
        return (
            f"Backbone: {cls.BACKBONE}, ImgSize: {cls.IMG_SIZE}, "
            f"BS: {cls.BATCH_SIZE}, LR: {cls.LEARNING_RATE}, "
            f"Epochs: {cls.EPOCHS}, Seed: {cls.SEED}"
        )
