import os
import torch


class Config:
    """
    Configuration class for the Iterative Attentive SWA-Distillation pipeline.
    Centralizes all hyperparameters, paths, and model settings.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SUBSET_SIZE = 20

    # =========================================================================
    # Paths
    # =========================================================================
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"

    # Source Data
    SPECTROGRAM_DIR = os.path.join(INPUT_ROOT, "supplemental_data", "spectrograms")
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    WORKING_DIR = "./working/idea_14"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Model Architecture
    # =========================================================================
    MODEL_NAME = "seresnet34"  # timm model name
    PRETRAINED = True
    NUM_CLASSES = 19

    # =========================================================================
    # Data Preprocessing
    # =========================================================================
    # High-Fidelity Alignment Resolution
    IMG_HEIGHT = 256
    IMG_WIDTH = 640
    IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

    # ImageNet Normalization Stats
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # Augmentation
    MIXUP_ALPHA = 0.2

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    EPOCHS = 50
    # Activate SWA for the final ~25% of epochs (Start at epoch 38, run for 12 epochs)
    SWA_START_EPOCH = 38
    SWA_LR = 1e-4  # Learning rate for SWA phase

    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Pipeline Settings
    NUM_TEACHERS = 3  # Number of independent teacher models to train in Stage 1

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Adjust based on available vCPUs (12 available)

    @classmethod
    def setup(cls):
        """
        Ensures necessary directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories upon module import
Config.setup()
