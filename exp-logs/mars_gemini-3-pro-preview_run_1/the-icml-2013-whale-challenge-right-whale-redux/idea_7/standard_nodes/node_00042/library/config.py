import os


class Config:
    """
    Configuration for the Right Whale Detection Task.
    Implements parameters for Multi-Resolution Time-Preserving ResNet-34 CRNN.
    """

    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # Audio Signal Processing
    # ==========================================
    SAMPLE_RATE = 2000
    DURATION = 2.0  # Fixed duration in seconds
    N_MELS = 128

    # Multi-Resolution Strategy
    # Window sizes in samples corresponding to different temporal resolutions
    # At 2000Hz: 200 (100ms), 500 (250ms), 1000 (500ms)
    WINDOW_SIZES = [200, 500, 1000]

    # Hop length in samples
    # 10ms hop -> 0.01s * 2000Hz = 20 samples
    HOP_LENGTH = 20

    # ==========================================
    # Data Paths
    # ==========================================
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"

    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for caching processed tensors (npy files)
    WORKING_DIR = "./working/idea_7"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Submission directory
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Model & Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 20

    # Class Imbalance Handling
    # Explicit positive class weight for BCEWithLogitsLoss
    POS_WEIGHT = 9.0

    # Augmentation
    MIXUP_ALPHA = 0.4

    # SpecAugment Constraints
    # Limit time masking to avoid occluding short whale calls
    MAX_TIME_MASK_MS = 200
    # Calculate max frames: 200ms / (10ms/frame) = 20 frames
    MAX_TIME_MASK_FRAMES = int(MAX_TIME_MASK_MS / (HOP_LENGTH / SAMPLE_RATE * 1000))

    # Hardware
    NUM_WORKERS = 4
