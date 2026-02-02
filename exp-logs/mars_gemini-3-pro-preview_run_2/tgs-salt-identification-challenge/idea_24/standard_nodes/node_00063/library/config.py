import os
import torch


class Config:
    """
    Global configuration for Multi-Task Wide-LinkNet with Ensemble Soft-Distillation.
    """

    # -------------------------------------------------------------------------
    # System & Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Adjust number of workers based on available vCPUs (12)
    NUM_WORKERS = 4

    # -------------------------------------------------------------------------
    # File Paths & Directories
    # -------------------------------------------------------------------------
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Experiment specific directory
    IDEA_NAME = "idea_optimized"
    IDEA_DIR = os.path.join(WORKING_DIR, IDEA_NAME)

    # Sub-directories for artifacts
    CACHE_DIR = os.path.join(IDEA_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(IDEA_DIR, "checkpoints")

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    DEPTHS_PATH = os.path.join(INPUT_ROOT, "depths.csv")

    # Final Submission
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Processing & Dimensions
    # -------------------------------------------------------------------------
    # Original image dimensions
    ORIG_H = 101
    ORIG_W = 101

    # Padded dimensions for network divisibility (32 * 4 = 128)
    IMG_H = 128
    IMG_W = 128

    # Input channels (1 for summed RGB/Grayscale)
    IN_CHANNELS = 1

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    BACKBONE = "resnet34"
    ENCODER_WEIGHTS = "imagenet"

    # Multi-task learning weight for auxiliary depth regression head
    AUX_DEPTH_LOSS_WEIGHT = 0.1

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    NUM_FOLDS = 5
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4

    # Stage 1: Supervised Training
    STAGE1_EPOCHS = 50

    EARLY_STOPPING_PATIENCE = 10

    # Debugging / Quick Run
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 50

    # -------------------------------------------------------------------------
    # Augmentation Parameters
    # -------------------------------------------------------------------------
    # Non-Rigid: Elastic Transform
    AUG_ELASTIC_ALPHA = 120
    AUG_ELASTIC_SIGMA = 6
    AUG_ELASTIC_ALPHA_AFFINE = 120 * 0.03

    # Rigid: ShiftScaleRotate
    AUG_SHIFT_SCALE_ROTATE_P = 0.2

    # -------------------------------------------------------------------------
    # Evaluation Metrics
    # -------------------------------------------------------------------------
    # IoU thresholds for mAP calculation
    IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    @staticmethod
    def setup():
        """
        Creates the necessary directory structure for the experiment.
        Should be called at the start of the pipeline.
        """
        dirs_to_create = [
            Config.IDEA_DIR,
            Config.CACHE_DIR,
            Config.CHECKPOINT_DIR,
            Config.WORKING_DIR,
        ]

        for d in dirs_to_create:
            os.makedirs(d, exist_ok=True)

        print(f"Configuration setup complete. Working directory: {Config.IDEA_DIR}")
