import os
import torch


class Config:
    """
    Configuration for Manifold-Regularized High-Fidelity SWA-Distillation.
    Defines all hyperparameters, file paths, and training settings.
    """

    # =========================================================================
    # 1. File Paths & Directories
    # =========================================================================
    INPUT_ROOT = "./input"
    ESSENTIAL_DATA = os.path.join(INPUT_ROOT, "essential_data")
    SUPPLEMENTAL_DATA = os.path.join(INPUT_ROOT, "supplemental_data")

    # Metadata (Pre-generated in previous steps)
    METADATA_DIR = "./metadata"
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Data Sources
    SPECTROGRAM_DIR = os.path.join(SUPPLEMENTAL_DATA, "spectrograms")

    # Working Directory (Artifacts for this specific idea)
    WORKING_DIR = "./working/idea_22"
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    LOG_DIR = os.path.join(WORKING_DIR, "logs")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # 2. Data Configuration
    # =========================================================================
    # High-Fidelity Resolution
    IMG_HEIGHT = 256
    IMG_WIDTH = 640

    # ImageNet Normalization Statistics
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # Dataset Properties
    NUM_CLASSES = 19
    NUM_WORKERS = 4  # Safe default for 12 vCPUs

    # =========================================================================
    # 3. Model Configuration
    # =========================================================================
    MODEL_ARCH = "resnet34"
    PRETRAINED = True

    # Input Mixup
    # Alpha for Beta distribution
    MIXUP_ALPHA = 0.2

    # =========================================================================
    # 4. Training Configuration
    # =========================================================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Default Hyperparameters
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # SWA (Stochastic Weight Averaging)
    SWA_LR = 1e-4

    # Distillation Ensemble
    NUM_TEACHERS = 3

    def __init__(self, debug=False, epochs=None, batch_size=None):
        """
        Initialize configuration with overrides for flexibility.

        Args:
            debug (bool): If True, reduces epochs and dataset size for debugging.
            epochs (int, optional): Override default number of epochs.
            batch_size (int, optional): Override default batch size.
        """
        self.debug = debug

        # Set Epochs
        if epochs is not None:
            self.epochs = epochs
        else:
            self.epochs = 2 if debug else self.DEFAULT_EPOCHS

        # Set Batch Size
        if batch_size is not None:
            self.batch_size = batch_size
        else:
            self.batch_size = 8 if debug else self.DEFAULT_BATCH_SIZE

        # SWA Schedule: Active for the last 25% of training
        # Example: 50 epochs -> Start SWA at epoch 38 (0-indexed 37)
        self.swa_start_epoch = int(self.epochs * 0.75)

        # Ensure directory structure exists
        self._setup_dirs()

    def _setup_dirs(self):
        """Creates necessary directories for the experiment."""
        for path in [
            self.WORKING_DIR,
            self.CHECKPOINT_DIR,
            self.CACHE_DIR,
            self.LOG_DIR,
            self.SUBMISSION_DIR,
        ]:
            os.makedirs(path, exist_ok=True)

    def get_transform_params(self):
        """Returns a dictionary of transformation parameters."""
        return {
            "height": self.IMG_HEIGHT,
            "width": self.IMG_WIDTH,
            "mean": self.MEAN,
            "std": self.STD,
        }

    def __repr__(self):
        return (
            f"Config(debug={self.debug}, epochs={self.epochs}, "
            f"batch_size={self.batch_size}, swa_start={self.swa_start_epoch}, "
            f"device={self.DEVICE})"
        )
