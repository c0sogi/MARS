import os
import torch


class Config:
    """
    Configuration for Idea 15: 2.5D U-Net++ with Dynamic Scale Training.
    """

    # Meta
    SEED = 42
    EXP_NAME = "idea_15"
    COMMENT = "2.5D U-Net++ with EfficientNet-B4 backbone and Dynamic Scale Training"
    DEBUG = False  # Set to True for fast debugging runs

    # Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = os.path.join("./working", EXP_NAME)

    # Metadata Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # Data / Image
    # 2.5D input: Slice i-1, Slice i, Slice i+1
    IN_CHANNELS = 3
    # Classes: large_bowel, small_bowel, stomach
    NUM_CLASSES = 3

    # Resolution
    # Max resolution for inference and max training scale
    IMG_SIZE = 512
    # Dynamic scales for training to address resolution bottleneck
    IMG_SCALES = [320, 384, 448, 512]

    # Model Architecture
    ARCH = "UnetPlusPlus"
    BACKBONE = "efficientnet-b4"
    ENCODER_WEIGHTS = "imagenet"

    # Training Hyperparameters
    BATCH_SIZE = 24  # Fits on A100 40GB with EffNet-B4 @ 512x512
    EPOCHS = 15 if not DEBUG else 2
    LR = 2e-4
    MIN_LR = 1e-6
    WEIGHT_DECAY = 1e-5

    # Optimizer & Scheduler
    SCHEDULER = "CosineAnnealingLR"
    T_MAX = EPOCHS  # For CosineAnnealing

    # Loss Function (BCE + Tversky)
    # Tversky beta > alpha to emphasize recall (penalize FN)
    TVERSKY_ALPHA = 0.3
    TVERSKY_BETA = 0.7
    TVERSKY_SMOOTH = 1.0
    # Weights for combined loss
    BCE_WEIGHT = 0.5
    TVERSKY_WEIGHT = 0.5

    # Inference / Post-processing
    THRESHOLD = 0.5
    MIN_PIXELS = 0  # Minimum pixels to keep a mask, can be tuned

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    @classmethod
    def setup(cls):
        """
        Ensures the working directory exists and sets reproducible seeds.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)

        # Set seeds for reproducibility
        import random
        import numpy as np

        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def __repr__(self):
        return f"<Config {self.EXP_NAME}: {self.COMMENT}>"
