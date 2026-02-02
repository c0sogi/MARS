import os
import torch


class Config:
    # ==========================================
    # General Settings
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging

    # ==========================================
    # Directory Paths
    # ==========================================
    # Input Directories
    INPUT_DIR = "./input"
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")
    METADATA_DIR = "./metadata"

    # Metadata Files (Generated previously)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Working Directories
    # Using 'idea_17' as the specific workspace for this high-density ensemble strategy
    WORKING_DIR = "./working/idea_17"
    OUTPUT_DIR = os.path.join(WORKING_DIR, "output")
    MODELS_DIR = os.path.join(WORKING_DIR, "models")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Submission Directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Configuration
    # ==========================================
    IMG_SIZE = 256
    NUM_WORKERS = 4  # Optimized for 12 vCPUs
    CLASSES = ["healthy", "multiple_diseases", "rust", "scab"]
    NUM_CLASSES = len(CLASSES)

    # ==========================================
    # Model & Training Hyperparameters
    # ==========================================
    MODEL_NAME = "resnet34"
    PRETRAINED = True

    # High-Density Stratified K-Fold
    N_FOLDS = 10
    N_SEEDS = 5

    # Training Loop
    EPOCHS = 15
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-6

    # Scheduler: Cosine Annealing Warm Restarts
    # Synchronized with EPOCHS as per strategy
    T_0 = 15  # Cycle length matches total epochs
    T_MULT = 1
    ETA_MIN = 1e-6

    # Loss & Optimization
    USE_CLASS_WEIGHTS = True  # To handle class imbalance
    INITIAL_LOSS_THRESHOLD = 1.38  # -ln(1/4), used for initialization safeguard

    # Inference
    USE_TTA = True  # Test Time Augmentation (Horizontal + Vertical Flip)

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def setup():
        """
        Creates necessary directories for the experiment.
        Ensures directory safety.
        """
        dirs_to_create = [
            Config.WORKING_DIR,
            Config.OUTPUT_DIR,
            Config.MODELS_DIR,
            Config.CACHE_DIR,
            Config.SUBMISSION_DIR,
        ]

        for d in dirs_to_create:
            os.makedirs(d, exist_ok=True)

        print(f"Directories initialized at {Config.WORKING_DIR}")
        print(f"Device set to: {Config.DEVICE}")


# Initialize directories upon import
Config.setup()
