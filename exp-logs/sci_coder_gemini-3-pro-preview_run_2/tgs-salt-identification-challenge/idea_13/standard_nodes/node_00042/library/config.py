import os
import torch


class Config:
    """
    Configuration for the Residual-Injection Wide-LinkNet Salt Segmentation task.
    Centralizes hyperparameters, file paths, and global constants.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Optimized for the available 12 vCPUs

    # =========================================================================
    # Data Paths & Directories
    # =========================================================================
    # Input (Read-Only)
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Files
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Output (Working Directory)
    WORKING_DIR = "./working/idea_13"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")

    # Specific Output Files
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # =========================================================================
    # Image & Preprocessing Parameters
    # =========================================================================
    ORIG_IMG_SIZE = 101
    IMG_SIZE = 128  # Padded size (multiple of 32 for ResNet stride)
    IN_CHANNELS = 1  # Summing RGB to 1 channel as per strategy

    # Normalization (ImageNet stats)
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # Scheduler
    T_MAX = 50  # For CosineAnnealingLR

    # =========================================================================
    # Model Architecture Parameters
    # =========================================================================
    BACKBONE = "resnet34"
    DEPTH_EMBEDDING_DIM = 32

    # =========================================================================
    # Augmentation & Regularization
    # =========================================================================
    AUG_PROB = 0.2

    # Elastic Transform Parameters
    ELASTIC_ALPHA = 120
    ELASTIC_SIGMA = 6

    # Depth Handling
    DEPTH_DROP_RATE = 0.2  # Probability to mask depth with 0 (Bernoulli masking) Cite solution_lesson_node_00029

    @classmethod
    def setup(cls):
        """
        Ensures all necessary working directories exist.
        Should be called at the start of the pipeline.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        print(f"Configuration setup complete. Working directory: {cls.WORKING_DIR}")
