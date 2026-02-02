import os
import torch


class Config:
    """
    Configuration class for Apple Disease Detection Task.
    Centralizes all hyperparameters, file paths, and model settings.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    PROJECT_NAME = "apple_disease_detection"
    IDEA_NAME = "idea_2"  # Strategy: ConvNeXt + 384px + GradAccum

    # Directories
    # Input is read-only
    INPUT_DIR = "./input"
    # Metadata is pre-generated
    METADATA_DIR = "./metadata"
    # Working directory for artifacts (checkpoints, logs, cache)
    WORKING_DIR = f"./working/{IDEA_NAME}"
    # Output directory for submission
    OUTPUT_DIR = "./submission"

    # Ensure writable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Random Seed for reproducibility
    SEED = 42

    # Compute
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Use available CPUs, but cap at reasonable number to avoid overhead
    NUM_WORKERS = min(12, os.cpu_count() if os.cpu_count() else 4)

    # =========================================================================
    # Data Configuration
    # =========================================================================
    # Paths to image directories
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    # Paths to metadata CSVs
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Submission output path
    SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")

    # Image parameters
    IMG_SIZE = 384  # High resolution for fine-grained disease detection
    NUM_CLASSES = 6

    # Class labels (Alphabetical order assumed for MultiLabelBinarizer)
    CLASSES = [
        "complex",
        "frog_eye_leaf_spot",
        "healthy",
        "powdery_mildew",
        "rust",
        "scab",
    ]

    # Debugging: Set to True to train/predict on a small subset
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100

    # =========================================================================
    # Model Configuration
    # =========================================================================
    # ConvNeXt Small with LayerNorm is more stable for small physical batches
    # compared to BN-heavy models like EfficientNet.
    # 'fb_in22k' indicates weights pretrained on ImageNet-22k.
    MODEL_NAME = "convnext_small.fb_in22k"
    PRETRAINED = True

    # Regularization inside model
    DROP_RATE = 0.0
    DROP_PATH_RATE = 0.1  # Stochastic depth rate

    # =========================================================================
    # Training Configuration
    # =========================================================================
    EPOCHS = 15

    # Gradient Accumulation Strategy
    # We aim for an Effective Batch Size of 32.
    # With IMG_SIZE=384, GPU memory might restrict Physical Batch Size.
    BATCH_SIZE = 16  # Physical batch size per forward/backward pass
    GRADIENT_ACCUM_STEPS = 2  # Accumulate gradients over 2 steps
    # Effective Batch Size = 16 * 2 = 32

    # Optimizer settings
    LEARNING_RATE = 2e-4
    MIN_LR = 1e-6
    WEIGHT_DECAY = 1e-4
    MAX_GRAD_NORM = 1000.0  # Gradient clipping

    # Scheduler settings (Cosine Annealing)
    WARMUP_EPOCHS = 1

    # =========================================================================
    # Inference Configuration
    # =========================================================================
    # Test Time Augmentation: Average predictions of original and h-flip
    USE_TTA = True

    # Threshold for multi-label classification (Sigmoid output)
    THRESHOLD = 0.5

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print("\n" + "=" * 40)
        print(f"CONFIGURATION: {cls.IDEA_NAME}")
        print("=" * 40)
        for key, value in cls.__dict__.items():
            if not key.startswith("__") and not callable(value):
                print(f"{key:<25} : {value}")
        print("=" * 40 + "\n")
