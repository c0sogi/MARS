import os
import torch


class Config:
    """
    Configuration module for Contrail Detection (Idea 4).
    Centralizes settings for the U-Net++ EfficientNet-B4 pipeline,
    including paths, model architecture, and training logic.
    """

    # --- General Settings ---
    SEED = 42
    IDEA_NAME = "idea_4"
    DEBUG = False  # Set to True to run on a small subset of data for debugging

    # --- File Paths ---
    INPUT_ROOT = "./input"
    METADATA_ROOT = "./metadata"

    # Directory for saving checkpoints, cache, and logs
    WORKING_DIR = os.path.join("./working", IDEA_NAME)

    # Metadata CSVs
    TRAIN_METADATA_PATH = os.path.join(METADATA_ROOT, "train.csv")
    VALIDATION_METADATA_PATH = os.path.join(METADATA_ROOT, "validation.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_ROOT, "test.csv")

    # Submission Output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Model Architecture ---
    # Standard U-Net with EfficientNet-B0 (Cite solution_lesson_node_00008)
    MODEL_ARCH = "Unet"
    BACKBONE = "efficientnet_b0"
    ENCODER_WEIGHTS = "imagenet"

    # Input: 3 channels (Ash Color) + 3 channels (Temporal Diff) = 6 Channels
    IN_CHANNELS = 6
    NUM_CLASSES = 1
    ACTIVATION = None  # Output raw logits (handled by loss/post-processing)

    # --- Data Configuration ---
    IMAGE_SIZE = 256

    # --- Training Hyperparameters ---
    # Batch size increased for B0 (Cite solution_lesson_node_00012)
    BATCH_SIZE = 64
    EPOCHS = 30  # Extended epochs (Cite solution_lesson_node_00009)

    # Optimizer: AdamW
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-2

    # Scheduler: Cosine Annealing
    T_MAX = 30  # Matches EPOCHS
    ETA_MIN = 1e-6

    # --- Advanced Training Strategies ---
    # Online Top-K Checkpointing: Keep only the best N models
    N_BEST_MODELS = 5

    # Convergence-Aware Weight Averaging:
    # Only average checkpoints saved *after* this epoch to ensure convergence
    CONVERGENCE_EPOCH_THRESHOLD = 20

    # --- Hardware & System ---
    NUM_WORKERS = 12  # Utilization of available vCPUs
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Creates the necessary working and submission directories.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize environment setup on import
Config.setup()
