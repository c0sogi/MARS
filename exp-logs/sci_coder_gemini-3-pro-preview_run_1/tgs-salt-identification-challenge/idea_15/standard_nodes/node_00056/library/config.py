import os
import torch


class Config:
    """
    Configuration for Deep Residual U-Net with Coordinate Attention and SWA-Lovasz Curriculum.
    """

    # -------------------------------------------------------------------------
    # 1. General Settings
    # -------------------------------------------------------------------------
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Adjust based on available vCPUs (12 available)

    # -------------------------------------------------------------------------
    # 2. File Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Experiment specific working directory
    WORKING_DIR = "./working/idea_15"
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")

    # Metadata files (Pre-generated)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Submission output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # 3. Data Parameters
    # -------------------------------------------------------------------------
    ORIG_SIZE = 101
    IMG_SIZE = 128  # Reflection padded size for training
    CHANNELS = 1  # Seismic image channels
    INPUT_CHANNELS = 2  # 1 Image + 1 Depth Map (Fused)

    # -------------------------------------------------------------------------
    # 4. Model Architecture
    # -------------------------------------------------------------------------
    # Deep Residual U-Net Config
    ENCODER_FILTERS = [64, 128, 256, 512]  # Avoid 1024 bottleneck
    DECODER_FILTERS = [256, 128, 64, 32]
    DEEP_SUPERVISION = True
    USE_SCSE = (
        True  # Cite solution_lesson_node_00055: scSE outperforms Coordinate Attention
    )

    # -------------------------------------------------------------------------
    # 5. Training Hyperparameters
    # -------------------------------------------------------------------------
    NUM_EPOCHS = 150  # Cite solution_lesson_node_00011: Extend training for convergence
    BATCH_SIZE = 64  # Safe for A100 40GB with 128x128 images
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Scheduler: Cosine Annealing Warm Restarts
    CYCLES = 3
    EPOCHS_PER_CYCLE = 50  # 3 cycles * 50 epochs = 150 total

    # -------------------------------------------------------------------------
    # 6. Curriculum & Strategy
    # -------------------------------------------------------------------------
    # Phase 1: Epochs 0-100 -> BCE + Dice
    # Phase 2: Epochs 101-150 -> BCE + Lovasz-Hinge
    LOVASZ_SWITCH_EPOCH = 100

    # Stochastic Weight Averaging (SWA)
    # Active during the end of the final cycle
    USE_SWA = True
    SWA_START_EPOCH = 130
    SWA_LR = 1e-4

    @classmethod
    def setup(cls):
        """
        Ensures strict directory safety for the experiment.
        Creates working and submission directories if they do not exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Execute setup on import to guarantee directory existence
Config.setup()
