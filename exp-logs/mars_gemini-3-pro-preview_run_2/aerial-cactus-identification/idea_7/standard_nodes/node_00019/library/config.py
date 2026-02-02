import os
import torch


class Config:
    """
    Global configuration for the Cactus Identification task.
    Contains file paths, hyperparameters, and model settings.
    """

    # ==========================================
    # Paths
    # ==========================================
    # Base directories
    INPUT_DIR = "./input"
    WORKING_DIR = "./working/idea_7"
    SUBMISSION_DIR = "./submission"

    # Metadata paths (pre-generated)
    TRAIN_METADATA = os.path.join("./metadata", "train_metadata.csv")
    VAL_METADATA = os.path.join("./metadata", "val_metadata.csv")
    TEST_METADATA = os.path.join("./metadata", "test_metadata.csv")

    # Image directories (relative to INPUT_DIR as per metadata)
    # Note: Metadata contains paths like 'train/id.jpg', so we join with INPUT_DIR

    # Output paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data & Preprocessing
    # ==========================================
    IMAGE_SIZE = (32, 32)
    NUM_CLASSES = 1
    NUM_WORKERS = 4

    # Normalization (ImageNet stats are often used, but for aerial cactus
    # simple 0-1 or mean/std of dataset is fine. We'll use standard 0.5 for simplicity
    # or dataset specific stats if computed. Here we stick to 0-1 normalization logic in dataset)

    # ==========================================
    # Model Architecture (Custom ResNet-UNet)
    # ==========================================
    # Encoder channel configuration [Stage1, Stage2, Stage3]
    ENCODER_CHANNELS = [16, 32, 64]

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    # Ensemble strategy: Homogeneous Seed Averaging
    SEEDS = [0, 1, 2, 3, 4]

    # Optimization
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2  # AdamW standard
    NUM_EPOCHS = 30

    # Scheduler (Cosine Annealing)
    T_MAX = NUM_EPOCHS  # For CosineAnnealingLR
    ETA_MIN = 1e-6

    # Early Stopping
    PATIENCE = 7

    # ==========================================
    # Compute & Debugging
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Set to True to run a quick check with a small subset of data
    DEBUG = False
    DEBUG_SAMPLES = 100

    @classmethod
    def setup(cls):
        """
        Create necessary working directories.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def get_model_path(cls, seed):
        """
        Returns the file path for saving the model checkpoint for a specific seed.
        """
        return os.path.join(cls.WORKING_DIR, f"model_seed_{seed}.pth")
