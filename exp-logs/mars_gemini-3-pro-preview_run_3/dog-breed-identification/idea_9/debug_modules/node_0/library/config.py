import os
import torch


class Config:
    """
    Configuration class for the Dog Breed Classification task.
    Centralizes settings for the Homogeneous Stratified Ensemble approach
    using ConvNeXt-Small with Manual Weight Averaging.
    """

    # ==========================================
    # Reproducibility & Debugging
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data for debugging
    DEBUG_SUBSET_SIZE = 500  # Number of samples to use when DEBUG is True

    # ==========================================
    # Directories & Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Specific working directory for this solution idea
    WORKING_DIR = "./working/idea_9"

    # Metadata file paths (pre-generated)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Submission output
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # Ensure the working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # ==========================================
    # Model Architecture
    # ==========================================
    # Backbone: ConvNeXt-Small pre-trained on ImageNet-22k and fine-tuned on 1k
    # Selected for optimal trade-off between capacity and overfitting risk.
    MODEL_NAME = "convnext_small.fb_in22k_ft_in1k"
    NUM_CLASSES = 120

    # ==========================================
    # Input Pipeline
    # ==========================================
    IMG_SIZE = 224  # Fixed resolution to avoid batch size reduction
    BATCH_SIZE = 64  # Optimized for A100 40GB
    NUM_WORKERS = 8  # Efficient data loading with 12 vCPUs available

    # ==========================================
    # Training Strategy
    # ==========================================
    N_FOLDS = 5
    EPOCHS = 30  # Total epochs for the fine-tuning phase

    # Two-Phase Transfer Learning
    # Phase 1: Linear Probing (Freeze backbone, train head)
    PHASE1_EPOCHS = 1
    PHASE1_LR = 1e-3

    # Phase 2: Fine-Tuning (Unfreeze all)
    LEARNING_RATE = 1e-5  # Conservative LR to preserve pre-trained features
    WEIGHT_DECAY = 1e-4  # Standard regularization for ConvNeXt
    MIN_LR = 1e-7  # For Cosine Annealing scheduler

    # ==========================================
    # Advanced Optimization
    # ==========================================
    # Manual Weight Averaging (SWA)
    # We save checkpoints for the last N epochs and average their weights
    # to create a "Model Soup" that improves generalization.
    SWA_EPOCHS = 5

    # ==========================================
    # Hardware
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def print_config(cls):
        """Prints the current configuration for verification."""
        print("\n" + "=" * 40)
        print(f"CONFIGURATION: {cls.MODEL_NAME}")
        print("=" * 40)
        print(f"Input Size    : {cls.IMG_SIZE}x{cls.IMG_SIZE}")
        print(f"Batch Size    : {cls.BATCH_SIZE}")
        print(f"Epochs        : {cls.EPOCHS} (Phase 2) + {cls.PHASE1_EPOCHS} (Phase 1)")
        print(f"Learning Rate : {cls.LEARNING_RATE}")
        print(f"Folds         : {cls.N_FOLDS}")
        print(f"SWA Epochs    : {cls.SWA_EPOCHS}")
        print(f"Device        : {cls.DEVICE}")
        print(f"Working Dir   : {cls.WORKING_DIR}")
        print("=" * 40 + "\n")
