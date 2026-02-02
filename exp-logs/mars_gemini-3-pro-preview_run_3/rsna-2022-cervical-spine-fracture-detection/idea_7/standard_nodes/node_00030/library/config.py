import os
import torch


class Config:
    """
    Configuration class for the Multi-Scale Contextualized Instance-MIL Network.
    Centralizes all hyperparameters, paths, and environment settings.
    """

    # --- Environment & Reproducibility ---
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Use 12 workers as specified in the environment description
    NUM_WORKERS = 12

    # --- Data Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Image Directories
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    # Metadata Files (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # --- Caching ---
    # Specific working directory for Idea Optimized (Standard Window + Simple Context)
    CACHE_DIR = "./working/idea_optimized"
    # Ensure cache directory exists immediately upon config load
    os.makedirs(CACHE_DIR, exist_ok=True)

    # --- Data Preprocessing ---
    IMAGE_SIZE = 256
    # Uniformly sample 64 slices per exam as per the "Idea" description
    N_SLICES = 64
    # 2.5D Input: 3 channels (slice z-1, z, z+1)
    IN_CHANS = 3

    # --- Model Architecture ---
    BACKBONE = "resnet18"
    N_CLASSES = 7  # C1, C2, C3, C4, C5, C6, C7

    # --- Training Hyperparameters ---
    # Batch size of 8 for stability with ResNet18 backbone
    BATCH_SIZE = 8
    EPOCHS = 10

    # Optimizer (AdamW)
    LR = 2e-4
    WEIGHT_DECAY = 1e-2

    # Scheduler (Decoupled Cosine Annealing)
    # T_max set to 1.5x epochs to prevent premature decay
    T_MAX_MULTIPLIER = 1.5
    MIN_LR = 1e-6

    # Gradient Clipping
    MAX_GRAD_NORM = 1000.0

    # --- Evaluation & Submission ---
    MODEL_SAVE_PATH = "./working/best_model.pth"
    SUBMISSION_PATH = "./working/submission.csv"

    # --- Debugging Control ---
    DEBUG = False

    @classmethod
    def setup(cls, debug=False, epochs=None, batch_size=None, n_slices=None):
        """
        Allows dynamic reconfiguration of the Config class.
        Useful for debugging runs or hyperparameter tuning.
        """
        if debug:
            cls.DEBUG = True
            cls.EPOCHS = 2
            # Reduce slice count for faster debugging iteration if needed,
            # though 64 is preferred to test memory limits.
            if n_slices is None:
                cls.N_SLICES = 32

        if epochs is not None:
            cls.EPOCHS = epochs

        if batch_size is not None:
            cls.BATCH_SIZE = batch_size

        if n_slices is not None:
            cls.N_SLICES = n_slices
