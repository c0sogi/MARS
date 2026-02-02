import os
import torch


class Config:
    """
    Configuration class for the Whale Species Prediction task.
    Centralizes all hyperparameters, file paths, and model settings.
    """

    # --------------------------------------------------------------------------
    # General Setup
    # --------------------------------------------------------------------------
    SEED = 42
    # Use CUDA if available, otherwise CPU
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Number of workers for data loading (adjust based on vCPUs)
    NUM_WORKERS = 4

    # --------------------------------------------------------------------------
    # File Paths & Directories
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Working directory for this specific idea/experiment
    WORKING_DIR = "./working/idea_4"
    SUBMISSION_DIR = "./submission"

    # Create necessary directories safely
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Submission Output Path
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Preprocessing & Caching
    # --------------------------------------------------------------------------
    # Input image resolution (Higher resolution for fine-grained details)
    IMAGE_SIZE = 384

    # Cache Filenames
    # We use parameterized names to ensure we don't load stale data from other resolutions
    CACHE_TRAIN_IMAGES = os.path.join(WORKING_DIR, "train_images_b4_384.npy")
    CACHE_VAL_IMAGES = os.path.join(WORKING_DIR, "val_images_b4_384.npy")
    CACHE_TEST_IMAGES = os.path.join(WORKING_DIR, "test_images_b4_384.npy")

    # --------------------------------------------------------------------------
    # Model Architecture
    # --------------------------------------------------------------------------
    # Backbone: EfficientNet-B4 (Good balance of capacity and compute for 384px)
    MODEL_NAME = "tf_efficientnet_b4"
    # Embedding dimensionality
    EMBEDDING_SIZE = 512
    # Pooling strategy
    USE_GEM_POOLING = True
    # Dropout rate for the head
    DROPOUT_RATE = 0.3
    # Load pretrained ImageNet weights
    PRETRAINED = True

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    # Batch size (Decoupled from negative mining, so can be moderate)
    BATCH_SIZE = 16
    # Number of training epochs
    NUM_EPOCHS = 25
    # Initial learning rate
    LEARNING_RATE = 3e-4
    # Weight decay for optimizer
    WEIGHT_DECAY = 1e-4

    # ArcFace Loss Hyperparameters
    MARGIN = 0.50
    SCALE = 30.0

    # Scheduler Settings (ReduceLROnPlateau or Cosine)
    SCHEDULER_PATIENCE = 2
    SCHEDULER_FACTOR = 0.5
    MIN_LR = 1e-6

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 6

    # Model Checkpoint Path
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # --------------------------------------------------------------------------
    # Inference & Post-Processing
    # --------------------------------------------------------------------------
    # Open-Set Rejection Threshold:
    # If the cosine similarity to the nearest known whale is below this, predict 'new_whale'
    NEW_WHALE_THRESHOLD = 0.45

    # Re-ranking Configuration (k-Reciprocal Encoding)
    USE_RERANKING = True
    RERANK_K1 = 20
    RERANK_K2 = 6
    RERANK_LAMBDA = 0.3

    # --------------------------------------------------------------------------
    # Debugging
    # --------------------------------------------------------------------------
    # If True, runs the pipeline on a small subset of data
    DEBUG = False
    # Number of samples to use if DEBUG is True
    DEBUG_SAMPLES = 100

    @classmethod
    def print_config(cls):
        """Helper to print the current configuration for logging."""
        print("\n" + "=" * 40)
        print("CONFIGURATION")
        print("=" * 40)
        for key, value in cls.__dict__.items():
            if not key.startswith("__") and not callable(value):
                print(f"{key:<25}: {value}")
        print("=" * 40 + "\n")
