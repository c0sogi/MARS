import os


class Config:
    """
    Configuration class for the Right Whale Detection task.
    Defines hyperparameters for data processing, model training, and file paths.
    """

    def __init__(
        self, debug: bool = False, num_epochs: int = 20, batch_size: int = 128
    ):
        """
        Initialize configuration with optional overrides for debugging and runtime control.

        Args:
            debug (bool): If True, enables debug mode (e.g., smaller dataset).
            num_epochs (int): Number of training epochs.
            batch_size (int): Batch size for training and inference.
        """
        # --- General ---
        self.SEED = 42
        self.DEBUG = debug
        self.NUM_WORKERS = 4  # Adjust based on vCPU count (12 available)

        # --- Audio / Spectrogram Parameters ---
        # "Golden Recipe" from Idea description
        self.SAMPLE_RATE = 2000
        self.N_FFT = 1024
        self.HOP_LENGTH = 64
        self.N_MELS = 128
        self.TOP_DB = 80.0
        self.FMIN = 50
        self.FMAX = 1000

        # Resulting Image Dimensions: (n_mels, time_steps)
        # Time steps ~= 2.0s * 2000 / 64 = 62.5 -> 63
        self.IMG_SIZE = (128, 63)

        # --- Model Parameters ---
        self.MODELS = ["tf_efficientnet_b0_ns", "resnet34"]
        self.IN_CHANNELS = 1
        self.NUM_CLASSES = 1  # Binary classification

        # --- Training Parameters ---
        self.NUM_EPOCHS = num_epochs
        self.BATCH_SIZE = batch_size
        self.LEARNING_RATE = 1e-4
        self.WEIGHT_DECAY = 1e-2
        self.NUM_FOLDS = 5

        # Augmentation
        self.FREQ_MASK_PARAM = 20
        self.TIME_MASK_PARAM = 10

        # Pseudo-labeling
        self.PSEUDO_LABEL_CONF_HIGH = 0.95
        self.PSEUDO_LABEL_CONF_LOW = 0.05

        # --- Paths ---
        self.INPUT_ROOT = "./input"
        self.TRAIN_DIR = os.path.join(self.INPUT_ROOT, "train2")
        self.TEST_DIR = os.path.join(self.INPUT_ROOT, "test2")

        self.METADATA_DIR = "./metadata"
        self.TRAIN_CSV = os.path.join(self.METADATA_DIR, "train.csv")
        self.VAL_CSV = os.path.join(self.METADATA_DIR, "val.csv")
        self.TEST_CSV = os.path.join(self.METADATA_DIR, "test.csv")

        # Working Directory for Idea 20
        self.WORKING_DIR = "./working/idea_20"
        self.CACHE_DIR = os.path.join(self.WORKING_DIR, "cache")
        self.CHECKPOINT_DIR = os.path.join(self.WORKING_DIR, "checkpoints")
        self.OOF_DIR = os.path.join(self.WORKING_DIR, "oof")
        self.PREDS_DIR = os.path.join(self.WORKING_DIR, "preds")

        # Submission
        self.SUBMISSION_DIR = "./submission"
        self.SUBMISSION_FILE = os.path.join(self.SUBMISSION_DIR, "submission.csv")
        self.SAMPLE_SUBMISSION = os.path.join(self.INPUT_ROOT, "sampleSubmission.csv")

    def create_directories(self):
        """Creates necessary output directories if they don't exist."""
        os.makedirs(self.WORKING_DIR, exist_ok=True)
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.OOF_DIR, exist_ok=True)
        os.makedirs(self.PREDS_DIR, exist_ok=True)
        os.makedirs(self.SUBMISSION_DIR, exist_ok=True)
        print(f"Directories created at {self.WORKING_DIR}")

    def __repr__(self):
        return (
            f"Config(debug={self.DEBUG}, epochs={self.NUM_EPOCHS}, "
            f"batch_size={self.BATCH_SIZE}, models={self.MODELS})"
        )
