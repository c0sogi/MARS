import os
import torch


class Config:
    # =========================================================================
    # Directory & File Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    # Working directory for this specific idea/experiment
    WORKING_DIR = "./working/idea_51"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # Metadata files (pre-generated)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # =========================================================================
    # Data Preprocessing & Augmentation
    # =========================================================================
    # Radiological Windowing (Lung Window)
    WINDOW_LEVEL = -600
    WINDOW_WIDTH = 1500

    # Image Dimensions
    IMG_SIZE = 260  # Native resolution for EfficientNet-B2
    NUM_SLICES = 3  # Anchor + 2 boundary slices

    # Feature Scaling
    TIME_SCALE = 0.01  # Scale relative time

    # =========================================================================
    # Model Architecture
    # =========================================================================
    BACKBONE = "tf_efficientnet_b2_ns"  # timm backbone
    PRETRAINED = True
    IN_CHANNELS = 1  # Grayscale (windowed) input per slice, or 3 if stacked?
    # Usually efficientnet expects 3. We will stack 3 slices as channels.

    # Latent Dimensions
    IMG_PROJ_DIM = 64  # Bottleneck projection for image features
    LATENT_DIM = 128  # Hidden layer dimension for MLPs

    # Regularization
    # Explicitly excluding dropout in Stream B as per "Context-Aware Visual Residual"
    DROPOUT_RATE = 0.0

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to limit dataset size for debugging
    DEBUG_SAMPLES = 50  # Number of samples to use in debug mode

    EPOCHS = 50
    BATCH_SIZE = 32  # Strictly >= 32
    NUM_WORKERS = 4

    # Optimization
    LR_BACKBONE = 1e-4  # Differential LR: Lower for backbone
    LR_HEAD = 1e-3  # Differential LR: Higher for heads
    WEIGHT_DECAY = 1e-2

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS  # Dynamic horizon linked to total epochs
    ETA_MIN = 1e-6

    # Early Stopping
    PATIENCE = 10

    # =========================================================================
    # Metric & Loss Constraints
    # =========================================================================
    SIGMA_MIN = 70.0  # The floor for confidence (standard deviation)
    MAX_ERROR = 1000.0  # Error threshold for metric calculation
    SQRT_2 = 1.41421356  # Constant for Laplace Log Likelihood

    # Device configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """Creates necessary working directories."""
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def get_transforms(cls):
        """
        Returns transforms.
        Note: Actual implementation usually depends on library (albumentations),
        but config defines the parameters.
        """
        pass
