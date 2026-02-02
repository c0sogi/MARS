import os
import torch


class Config:
    """
    Configuration for Multi-Generational High-Fidelity ResNet-34 Distillation with SWA.
    Idea: 19
    """

    # --------------------------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------------------------
    SEED = 42

    # --------------------------------------------------------------------------
    # File Systems & Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Files
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Data Source
    SPECTROGRAM_DIR = os.path.join(INPUT_DIR, "supplemental_data", "spectrograms")

    # Working Directory (Idea 19)
    WORKING_DIR = "./working/idea_19"
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    LOG_DIR = os.path.join(WORKING_DIR, "logs")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Preprocessing
    # --------------------------------------------------------------------------
    # High-Fidelity Resolution: Preserves frequency resolution and temporal morphology
    IMG_HEIGHT = 256
    IMG_WIDTH = 640

    # ImageNet Normalization (Required for Pretrained Weights)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # --------------------------------------------------------------------------
    # Model Architecture
    # --------------------------------------------------------------------------
    MODEL_NAME = "resnet34"
    NUM_CLASSES = 19
    PRETRAINED = True

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 50

    # SWA (Stochastic Weight Averaging) Configuration
    # Activate SWA for the final 25% of epochs
    SWA_START_EPOCH = int(EPOCHS * 0.75)
    SWA_LR = 1e-4

    # Regularization
    MIXUP_ALPHA = 0.2

    # --------------------------------------------------------------------------
    # Hardware & Execution
    # --------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Optimized for 12 vCPUs

    @classmethod
    def setup(cls):
        """Creates the necessary working directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
