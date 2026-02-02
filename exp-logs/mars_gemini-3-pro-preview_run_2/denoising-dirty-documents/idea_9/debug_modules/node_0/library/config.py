import os
import torch


class Config:
    """
    Configuration class for the Coordinate Res2Net U-Net (CoRes2Net-UNet) denoising task.
    Encapsulates all hyperparameters, file paths, and runtime settings.
    """

    def __init__(
        self, debug: bool = False, num_epochs: int = 100, batch_size: int = 32
    ):
        """
        Initialize the configuration with optional overrides for debugging and tuning.

        Args:
            debug (bool): If True, runs in debug mode with reduced epochs and dataset size.
            num_epochs (int): Total number of training epochs.
            batch_size (int): Batch size for training.
        """
        # Reproducibility
        self.SEED = 42

        # Paths
        self.METADATA_DIR = "./metadata"
        self.TRAIN_METADATA = os.path.join(self.METADATA_DIR, "train.csv")
        self.VAL_METADATA = os.path.join(self.METADATA_DIR, "val.csv")
        self.TEST_METADATA = os.path.join(self.METADATA_DIR, "test.csv")

        self.INPUT_DIR = "./input"

        # Working directory for Idea 9 (CoRes2Net-UNet)
        self.WORKING_DIR = "./working/idea_9"
        os.makedirs(self.WORKING_DIR, exist_ok=True)

        self.CHECKPOINT_PATH = os.path.join(self.WORKING_DIR, "cores2net_best.pth")

        # Submission directory
        self.SUBMISSION_DIR = "./submission"
        os.makedirs(self.SUBMISSION_DIR, exist_ok=True)
        self.SUBMISSION_PATH = os.path.join(self.SUBMISSION_DIR, "submission.csv")

        # Data Parameters
        self.PATCH_SIZE = 128
        # High density sampling: 100 patches per image per epoch
        # In debug mode, we drastically reduce this for speed
        self.PATCHES_PER_IMAGE = 10 if debug else 100
        self.NUM_WORKERS = 4

        # Model Parameters
        self.IN_CHANNELS = 1
        self.OUT_CHANNELS = 1
        self.BASE_FILTERS = 64

        # Training Hyperparameters
        self.BATCH_SIZE = batch_size
        self.NUM_EPOCHS = 2 if debug else num_epochs
        self.LEARNING_RATE = 1e-3
        self.WEIGHT_DECAY = 1e-2  # Aggressive weight decay for AdamW

        # Scheduler Parameters (Cosine Annealing)
        self.T_MAX = self.NUM_EPOCHS
        self.ETA_MIN = 1e-6

        # Hardware
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # Inference
        self.OVERLAP_RATIO = 0.5
        self.TTA_ENABLED = True

        # Debug Flag
        self.DEBUG = debug

    def print_config(self):
        """
        Prints the current configuration settings.
        """
        print("=" * 30)
        print("Configuration:")
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")
        print("=" * 30)
