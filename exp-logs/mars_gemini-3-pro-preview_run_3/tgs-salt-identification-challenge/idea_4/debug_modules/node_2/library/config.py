import os
import torch


class Config:
    """
    Configuration class for Salt Segmentation Task.
    """

    def __init__(self, debug: bool = False, epochs: int = None):
        # --- General Settings ---
        self.SEED = 42
        self.DEBUG = debug

        # --- Compute ---
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.NUM_WORKERS = 4

        # --- Paths ---
        self.INPUT_DIR = "./input"
        self.METADATA_DIR = "./metadata"
        self.WORKING_DIR = "./working"

        # Data Paths
        self.TRAIN_METADATA_PATH = os.path.join(self.METADATA_DIR, "train_metadata.csv")
        self.VAL_METADATA_PATH = os.path.join(self.METADATA_DIR, "val_metadata.csv")
        self.TEST_METADATA_PATH = os.path.join(self.METADATA_DIR, "test_metadata.csv")

        self.TRAIN_IMAGES_DIR = os.path.join(self.INPUT_DIR, "train", "images")
        self.TRAIN_MASKS_DIR = os.path.join(self.INPUT_DIR, "train", "masks")
        self.TEST_IMAGES_DIR = os.path.join(self.INPUT_DIR, "test", "images")
        self.DEPTHS_CSV = os.path.join(self.INPUT_DIR, "depths.csv")

        # Output Paths
        self.IDEA_NAME = "idea_4"
        self.OUTPUT_DIR = os.path.join(self.WORKING_DIR, self.IDEA_NAME)
        self.CHECKPOINT_DIR = os.path.join(self.OUTPUT_DIR, "checkpoints")
        self.CACHE_DIR = self.OUTPUT_DIR  # For caching processed numpy arrays

        self.SUBMISSION_DIR = "./submission"
        self.SUBMISSION_PATH = os.path.join(self.SUBMISSION_DIR, "submission.csv")

        # --- Model Hyperparameters ---
        self.ENCODER_NAME = "resnext50_32x4d"
        self.ENCODER_WEIGHTS = "imagenet"
        self.IN_CHANNELS = 4  # 3 (RGB) + 1 (Depth)
        self.CLASSES = 1
        self.ACTIVATION = "sigmoid"

        # --- Data Preprocessing ---
        self.IMG_SIZE_ORIG = 101
        self.IMG_SIZE_MODEL = 128  # Pad to 128x128 for U-Net

        # --- Training Hyperparameters ---
        self.BATCH_SIZE = 32
        self.LEARNING_RATE = 1e-4
        self.WEIGHT_DECAY = 1e-4

        # Loss Schedule
        # Epoch to switch from BCE+Dice to Lovasz-Hinge
        self.LOVASZ_SWITCH_EPOCH = 15

        if epochs is not None:
            self.EPOCHS = epochs
        else:
            self.EPOCHS = 50

        self.EARLY_STOPPING_PATIENCE = 10

        # --- Debug Overrides ---
        if self.DEBUG:
            self.EPOCHS = 2
            self.BATCH_SIZE = 8
            self.NUM_WORKERS = 0  # Easier debugging
            print(
                f"Debug mode enabled: EPOCHS={self.EPOCHS}, BATCH_SIZE={self.BATCH_SIZE}"
            )

        # --- Directory Setup ---
        self._create_dirs()

    def _create_dirs(self):
        """Creates necessary directories if they don't exist."""
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        os.makedirs(self.SUBMISSION_DIR, exist_ok=True)

    def get_model_save_path(self, filename="best_model.pth"):
        return os.path.join(self.CHECKPOINT_DIR, filename)
