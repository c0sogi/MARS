import os
import torch


class Config:
    """
    Configuration class for the Digital Pathology Tumor Detection Task.
    Implements settings for Idea 2: DenseNet121 with Hard Attention and Conservative Augmentation.
    """

    # ==========================================
    # Project Structure & Paths
    # ==========================================
    PROJECT_NAME = "idea_2"
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = f"./working/{PROJECT_NAME}"
    SUBMISSION_DIR = "./submission"

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Parameters
    # ==========================================
    # Original image dimensions are 96x96
    ORIGINAL_SIZE = 96

    # Hard Attention: Crop center 48x48 to focus on ROI + context margin
    # This aligns with the 32x32 GT region while removing background noise
    INPUT_SIZE = 48

    # Binary classification (Tumor vs No Tumor)
    NUM_CLASSES = 1

    # ==========================================
    # Model Architecture
    # ==========================================
    # Using DenseNet121 for feature reuse and efficiency
    # Note: Implementation will require modifying the input stem for small resolution
    MODEL_ARCH = "densenet121"

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 1e-4

    # Relaxed Early Stopping to handle volatile training dynamics
    PATIENCE = 6

    # Optimizer & Scheduler settings
    WEIGHT_DECAY = 1e-5
    MIN_LR = 1e-6  # For Cosine Annealing

    # ==========================================
    # Augmentation Strategy
    # ==========================================
    # Conservative augmentation to preserve stain biomarkers
    AUGMENTATION_PARAMS = {
        "horizontal_flip_prob": 0.5,
        "vertical_flip_prob": 0.5,
        "rotate_90_prob": 0.5,
        # Weak color jitter to avoid distorting tissue characteristics
        "brightness_limit": 0.1,
        "contrast_limit": 0.1,
    }

    # Test Time Augmentation (TTA)
    # 4 views: Original, H-Flip, V-Flip, Rot90
    TTA_STEPS = 4

    # ==========================================
    # Compute & Debugging
    # ==========================================
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debugging flags to control dataset size for rapid testing
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 1000

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print("=" * 30)
        print(f"Configuration: {cls.PROJECT_NAME}")
        print("=" * 30)
        print(f"Device: {cls.DEVICE}")
        print(f"Model: {cls.MODEL_ARCH}")
        print(f"Input Size: {cls.INPUT_SIZE}x{cls.INPUT_SIZE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Epochs: {cls.EPOCHS}")
        print(f"Patience: {cls.PATIENCE}")
        print(f"Debug Mode: {cls.DEBUG}")
        print("=" * 30)
