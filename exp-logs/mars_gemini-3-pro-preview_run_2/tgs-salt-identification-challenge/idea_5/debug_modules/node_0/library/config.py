import os


class Config:
    """
    Configuration for Salt Segmentation Task using Depth-Conditioned ResNet34-LinkNet.
    """

    # ==========================
    # General Settings
    # ==========================
    SEED = 42
    DEVICE = "cuda"
    NUM_WORKERS = 12  # Utilizing available vCPUs

    # ==========================
    # File Paths & Directories
    # ==========================
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_5"
    SUBMISSION_DIR = "./submission"

    # Input Files (using generated metadata)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    DEPTHS_CSV = os.path.join(INPUT_ROOT, "depths.csv")

    # Output Paths
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure mutable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================
    # Data Configuration
    # ==========================
    ORIG_IMG_SIZE = 101
    IMG_SIZE = 128  # Padded size (multiple of 32 for ResNet)
    CHANNELS = 1  # Grayscale input

    # Debugging/Testing
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use when DEBUG is True

    # ==========================
    # Model Architecture
    # ==========================
    ENCODER = "resnet34"
    DECODER = "linknet"
    ENCODER_WEIGHTS = "imagenet"

    # Depth Injection Settings
    USE_DEPTH = True
    DEPTH_EMBEDDING_DIM = 16  # Dimension to project scalar depth into

    # ==========================
    # Training Hyperparameters
    # ==========================
    N_FOLDS = 5
    EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 10
    EARLY_STOPPING_MIN_DELTA = 1e-4

    # ==========================
    # Loss & Metric
    # ==========================
    LOSS_TYPE = "bce_dice"  # Binary Cross Entropy + Dice Loss
    IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    # ==========================
    # Augmentation Parameters
    # ==========================
    # General probability for rigid transforms
    AUG_PROB = 0.5

    # ShiftScaleRotate
    SHIFT_LIMIT = 0.0625
    SCALE_LIMIT = 0.1
    ROTATE_LIMIT = 15

    # Elastic Transform (Non-rigid)
    # Tuned for physical significance of salt bodies
    ELASTIC_PROB = 0.2
    ELASTIC_ALPHA = 120
    ELASTIC_SIGMA = 6
    ELASTIC_ALPHA_AFFINE = 120 * 0.03

    # ==========================
    # Semi-Supervised Learning
    # ==========================
    PSEUDO_LABELING = True
    RETRAIN_EPOCHS = 50  # Epochs for the retraining phase
