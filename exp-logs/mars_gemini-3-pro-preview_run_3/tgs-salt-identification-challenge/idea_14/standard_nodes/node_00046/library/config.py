import os
import torch


class Config:
    """
    Central configuration for the Salt Segmentation Task.
    Implements the 'Stabilized Semi-Supervised U-Net++ Ensemble' strategy parameters.
    """

    # --------------------
    # General Configuration
    # --------------------
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Optimized for the 12 vCPU environment

    # --------------------
    # File System & Paths
    # --------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific strategy (Idea 14)
    # Used for caching processed data and saving intermediate states
    WORKING_DIR = "./working/idea_14"

    # Checkpoint storage
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")

    # Submission output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Pre-generated Metadata Paths
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    # --------------------
    # Data Processing
    # --------------------
    ORIG_SIZE = 101
    IMG_SIZE = 128  # Padded size for U-Net architecture (divisible by 32)
    CHANNELS = 3  # Input Multiplexing: [Seismic, Seismic, Depth]

    # --------------------
    # Model Architecture
    # --------------------
    # U-Net++ with ResNeXt-50 (32x4d) encoder and scSE attention
    ARCH = "UnetPlusPlus"
    ENCODER = "resnext50_32x4d"
    ENCODER_WEIGHTS = "imagenet"

    # Lightweight decoder channels as specified: [256, 128, 64, 32, 16]
    # This acts as a structural regularizer.
    DECODER_CHANNELS = (256, 128, 64, 32, 16)

    # Attention mechanism
    ATTENTION_TYPE = "scse"

    # --------------------
    # Training Hyperparameters
    # --------------------
    FOLDS = 5
    EPOCHS = 50  # Optimized for 2h runtime while ensuring convergence Cite solution_lesson_node_00020
    BATCH_SIZE = 64

    # Optimizer settings
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4

    # Loss Curriculum
    # Epoch 1-15: BCE + Dice (Warm-up)
    # Epoch 16-80: Lovasz-Hinge (Fine-tuning)
    LOSS_SWITCH_EPOCH = 16

    # --------------------
    # Setup Logic
    # --------------------
    @classmethod
    def setup(cls):
        """
        Ensures all necessary working directories exist.
        Must be called at the start of the pipeline.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories upon module import
Config.setup()
