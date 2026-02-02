import os
import torch


class Config:
    """
    Central configuration for the Deep Residual Denoising Network (ResDnCNN) pipeline.
    Handles file paths, hyperparameters, and environment settings.
    """

    # =========================================================================
    # 1. File Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TRAIN_CLEANED_DIR = os.path.join(INPUT_DIR, "train_cleaned")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata Paths (already generated)
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Working Directory for Caching and Checkpoints
    # Idea 8: ResDnCNN with Geometric Self-Ensemble
    WORKING_DIR = "./working/idea_8"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Cache Files for Processed Patches
    # Using .npy for efficient storage of high-density patches
    TRAIN_PATCHES_PATH = os.path.join(WORKING_DIR, "train_patches.npy")
    TRAIN_TARGETS_PATH = os.path.join(WORKING_DIR, "train_targets.npy")
    VAL_PATCHES_PATH = os.path.join(WORKING_DIR, "val_patches.npy")
    VAL_TARGETS_PATH = os.path.join(WORKING_DIR, "val_targets.npy")

    # Model Checkpoint
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "resdncnn_best_model.pth")

    # Final Submission
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # 2. Data Processing Hyperparameters
    # =========================================================================
    PATCH_SIZE = 50
    STRIDE = 10  # Low stride for high-density patch extraction

    # Data Augmentation (Flips and Rotations)
    AUGMENTATION = True

    # Normalization
    PIXEL_MIN = 0.0
    PIXEL_MAX = 1.0

    # =========================================================================
    # 3. Model Architecture (ResDnCNN)
    # =========================================================================
    IN_CHANNELS = 1
    NUM_FEATURES = 64
    # Depth: 16 Residual Blocks results in a network depth of ~34 layers
    # (Input + 16*2 + Output)
    NUM_RES_BLOCKS = 16
    KERNEL_SIZE = 3
    PADDING = 1  # To maintain spatial dimensions (no pooling)

    # =========================================================================
    # 4. Training Configuration
    # =========================================================================
    SEED = 42

    # A100 GPU allows for larger batch sizes even with unoptimized code
    BATCH_SIZE = 128

    # Training Duration
    # 24h limit allows for substantial training.
    # With stride 10, dataset is large, so 50 epochs is significant.
    NUM_EPOCHS = 50

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Scheduler: Cosine Annealing
    COSINE_T_MAX = NUM_EPOCHS
    COSINE_ETA_MIN = 1e-6

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 8

    # =========================================================================
    # 5. Inference Configuration
    # =========================================================================
    # Geometric Self-Ensemble (Test-Time Augmentation)
    # 8 transforms: Identity, Rot90, Rot180, Rot270, and their horizontal flips
    TTA_ENABLED = True

    # =========================================================================
    # 6. Hardware
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # 12 vCPUs available
    NUM_WORKERS = 4

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print("\n==== Configuration ====")
        print(f"Device: {cls.DEVICE}")
        print(f"Working Dir: {cls.WORKING_DIR}")
        print(f"Patch Size: {cls.PATCH_SIZE}, Stride: {cls.STRIDE}")
        print(
            f"Model: ResDnCNN (Blocks: {cls.NUM_RES_BLOCKS}, Feat: {cls.NUM_FEATURES})"
        )
        print(
            f"Training: {cls.NUM_EPOCHS} Epochs, BS={cls.BATCH_SIZE}, LR={cls.LEARNING_RATE}"
        )
        print(f"TTA Enabled: {cls.TTA_ENABLED}")
        print("=======================\n")
