import os
import torch


class Config:
    # Reproducibility
    SEED = 42

    # Data Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_5"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_PATH = "./submission/submission.csv"

    # Metadata Files
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")
    DEPTHS_CSV = os.path.join(INPUT_DIR, "depths.csv")

    # Image Parameters
    ORIG_SIZE = 101
    IMG_SIZE = 128  # Padded size for U-Net architecture requirements (divisible by 32)
    IN_CHANNELS = 3  # [Gray, Gray, Depth] - Cite solution_lesson_node_00014

    # Model Architecture
    ENCODER_NAME = "resnext50_32x4d"
    ENCODER_WEIGHTS = "imagenet"
    DEEP_SUPERVISION = True
    DECODER_ATTENTION_TYPE = "scse"  # scSE attention in decoder

    # Training Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-4

    # Scheduler (ReduceLROnPlateau)
    SCHEDULER_PATIENCE = 5
    SCHEDULER_FACTOR = 0.5
    MIN_LR = 1e-6

    # Loss Scheduling
    # Epoch to switch from BCE+Dice to Lovasz-Hinge
    LOVASZ_EPOCH_START = 15

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    @classmethod
    def setup(cls):
        """Creates necessary directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(cls.SUBMISSION_PATH), exist_ok=True)


# Initialize directories immediately upon import
Config.setup()
