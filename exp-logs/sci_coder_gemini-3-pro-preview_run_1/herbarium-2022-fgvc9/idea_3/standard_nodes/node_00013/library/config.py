import os
import random
import numpy as np
import torch


class Config:
    """
    Configuration class for the Hierarchical ConvNeXt-Base Plant Classification Task.
    """

    # ==========================================
    # Paths and Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata files (Pre-generated)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Taxonomy Metadata
    TRAIN_METADATA_JSON = os.path.join(INPUT_DIR, "train_metadata.json")

    # Output Directories
    WORKING_DIR = "./working/idea_3"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Model Architecture
    # ==========================================
    MODEL_NAME = "convnext_base.fb_in1k"
    IMG_SIZE = 256

    # Class Counts (Derived from Data Analysis)
    NUM_CLASSES = 15501  # Species
    NUM_GENERA = 2564  # Genus
    NUM_FAMILIES = 272  # Family

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32  # Conservative for A100 40GB with ConvNeXt-Base
    NUM_WORKERS = 12  # Available vCPUs

    # Optimization
    LR_BACKBONE = 1e-5  # Lower rate for pre-trained backbone
    LR_HEAD = 1e-3  # Higher rate for new classification heads
    WEIGHT_DECAY = 1e-4
    LABEL_SMOOTHING = 0.1

    # Multi-task Loss Weight
    LAMBDA_AUX = 1.0  # Weight for auxiliary (Genus/Family) losses

    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    def __init__(self, debug=False, num_epochs=10, batch_size=None):
        """
        Initialize configuration with optional overrides for debugging and tuning.

        Args:
            debug (bool): If True, enables debug mode (smaller dataset).
            num_epochs (int): Number of training epochs.
            batch_size (int, optional): Override default batch size.
        """
        self.debug = debug
        self.NUM_EPOCHS = num_epochs

        if batch_size is not None:
            self.BATCH_SIZE = batch_size

        # If debug is True, we will sample a small subset of data
        self.SAMPLE_SIZE = 5000 if debug else None

        # Ensure working directories exist
        os.makedirs(self.WORKING_DIR, exist_ok=True)
        os.makedirs(self.SUBMISSION_DIR, exist_ok=True)

        # Set seed immediately upon initialization
        self.seed_everything(self.SEED)

    @staticmethod
    def seed_everything(seed=42):
        """
        Sets the random seed for all relevant libraries to ensure reproducibility.
        """
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
