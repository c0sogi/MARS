import os
import torch


class Config:
    """
    Configuration class for the Denoising Project (Idea 5: RDN with High-Density Patching).
    """

    # --- General ---
    SEED = 42

    # --- Paths ---
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TRAIN_CLEANED_DIR = os.path.join(INPUT_DIR, "train_cleaned")
    TEST_DIR = os.path.join(INPUT_DIR, "test")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sampleSubmission.csv")

    METADATA_DIR = "./metadata"
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Working Directory for Idea 5 (Cache & Model Checkpoints)
    WORKING_DIR = "./working/idea_5"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Submission Directory
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Model Output Path
    MODEL_PATH = os.path.join(WORKING_DIR, "rdn_model.pth")

    # Data Cache Paths (for deterministic data processing)
    TRAIN_PATCHES_CACHE = os.path.join(WORKING_DIR, "train_patches.npy")
    VAL_PATCHES_CACHE = os.path.join(WORKING_DIR, "val_patches.npy")

    # --- Data Hyperparameters ---
    PATCH_SIZE = 50
    STRIDE = 10

    # --- Model Hyperparameters (RDN) ---
    # G0: Growth rate (number of feature maps in dense layers)
    RDN_G0 = 64
    # Number of Residual Dense Blocks (RDB)
    RDN_NUM_BLOCKS = 8
    # Number of convolutional layers per RDB
    RDN_NUM_LAYERS = 6
    # Kernel size for convolutions
    RDN_KERNEL_SIZE = 3
    # Number of image channels (1 for grayscale)
    CHANNELS = 1

    # --- Training Hyperparameters ---
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 50
    NUM_WORKERS = 4  # Number of subprocesses for data loading

    # --- Hardware ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def display(cls):
        """Prints the current configuration."""
        print("=" * 40)
        print(f"Configuration ({cls.DEVICE})")
        print("=" * 40)
        for attribute, value in cls.__dict__.items():
            if not attribute.startswith("__") and not callable(value):
                print(f"{attribute:<25} : {value}")
        print("=" * 40)
