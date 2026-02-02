import os
import torch


class Config:
    """
    Configuration class for the Ship vs. Iceberg classification task.
    Implements the settings for the Semi-Supervised ResNet-18 SWA Ensemble.
    """

    # --------------------------------------------------------------------------
    # Directory and File Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching processed data and checkpoints
    # Using 'idea_14' as the specific experiment identifier
    WORKING_DIR = "./working/idea_14"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Raw Data Files
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files (Pre-generated)
    TRAIN_META = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Cache Files (Numpy format for fast loading)
    CACHE_TRAIN_IMAGES = os.path.join(WORKING_DIR, "train_images.npy")
    CACHE_TRAIN_LABELS = os.path.join(WORKING_DIR, "train_labels.npy")
    CACHE_TRAIN_ANGLES = os.path.join(WORKING_DIR, "train_angles.npy")

    CACHE_TEST_IMAGES = os.path.join(WORKING_DIR, "test_images.npy")
    CACHE_TEST_IDS = os.path.join(WORKING_DIR, "test_ids.npy")
    CACHE_TEST_ANGLES = os.path.join(WORKING_DIR, "test_angles.npy")

    # Checkpoint Directory
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Submission Output
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Statistics (Global Min-Max from Data Analysis)
    # --------------------------------------------------------------------------
    # Used for Global Min-Max Normalization
    BAND_1_MIN = -45.5944
    BAND_1_MAX = 32.1806
    BAND_2_MIN = -45.6555
    BAND_2_MAX = 17.8628

    # --------------------------------------------------------------------------
    # Image Processing Parameters
    # --------------------------------------------------------------------------
    ORIGINAL_HEIGHT = 75
    ORIGINAL_WIDTH = 75

    # Upsampling target dimensions (ResNet standard)
    RESIZE_HEIGHT = 224
    RESIZE_WIDTH = 224

    # Input Channels: Band 1 (Norm), Band 2 (Norm), Average (Norm)
    IN_CHANNELS = 3

    # --------------------------------------------------------------------------
    # Model Hyperparameters
    # --------------------------------------------------------------------------
    MODEL_NAME = "resnet18"
    NUM_CLASSES = 1
    DROPOUT_RATE = 0.5
    USE_PRETRAINED = True

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    SEED = 42
    BATCH_SIZE = 32
    NUM_WORKERS = 4  # Leveraging available vCPUs

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.01
    LABEL_SMOOTHING = 0.05

    # Cross-Validation
    N_FOLDS = 5

    # Stochastic Weight Averaging (SWA) Schedule
    # The model trains normally for SWA_START_EPOCH, then SWA is applied for SWA_DURATION
    SWA_START_EPOCH = 30
    SWA_DURATION = 15
    TOTAL_EPOCHS = SWA_START_EPOCH + SWA_DURATION

    # --------------------------------------------------------------------------
    # Semi-Supervised Learning (Cycle 2)
    # --------------------------------------------------------------------------
    # Thresholds for selecting pseudo-labels from the Teacher Ensemble
    CONFIDENCE_THRESHOLD_HIGH = 0.95  # Confident Iceberg
    CONFIDENCE_THRESHOLD_LOW = 0.05  # Confident Ship

    # --------------------------------------------------------------------------
    # Hardware
    # --------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
