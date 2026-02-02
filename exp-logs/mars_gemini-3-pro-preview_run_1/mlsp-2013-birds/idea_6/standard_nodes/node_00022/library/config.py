import os


class Config:
    """
    Central configuration for the Bird Species Classification Task.
    Implements the settings for the Spectral-Dynamic ResNet-34 with Self-Distillation strategy.
    """

    # ==========================================
    # Directories and File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_6"

    # Ensure the working directory exists for caching and checkpoints
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Source Data Paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SPECTROGRAM_DIR = os.path.join(INPUT_DIR, "supplemental_data", "spectrograms")

    # Submission Path
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Caching Paths (for deterministic loading)
    # ==========================================
    # We use .npy for processed tensors/arrays to avoid re-computing deltas and resizing
    CACHE_TRAIN_IMAGES = os.path.join(WORKING_DIR, "train_images.npy")
    CACHE_TRAIN_LABELS = os.path.join(WORKING_DIR, "train_labels.npy")

    CACHE_VAL_IMAGES = os.path.join(WORKING_DIR, "val_images.npy")
    CACHE_VAL_LABELS = os.path.join(WORKING_DIR, "val_labels.npy")

    CACHE_TEST_IMAGES = os.path.join(WORKING_DIR, "test_images.npy")
    CACHE_TEST_IDS = os.path.join(WORKING_DIR, "test_ids.npy")

    # ==========================================
    # Model Checkpoints
    # ==========================================
    TEACHER_MODEL_PATH = os.path.join(WORKING_DIR, "teacher_resnet34.pth")
    STUDENT_MODEL_PATH = os.path.join(WORKING_DIR, "student_resnet34.pth")

    # ==========================================
    # Data Preprocessing Hyperparameters
    # ==========================================
    # Image Dimensions:
    # Height 256 preserves frequency resolution.
    # Width 512 captures temporal dynamics while densifying the signal.
    IMG_HEIGHT = 256
    IMG_WIDTH = 512

    # Input Channels: 3 (1: Intensity, 2: Delta, 3: Delta-Delta)
    IN_CHANNELS = 3

    NUM_CLASSES = 19

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42

    # Hardware settings
    NUM_WORKERS = 2
    BATCH_SIZE = 16  # Adjusted for 12GB+ VRAM with ResNet34

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Augmentation
    MIXUP_ALPHA = 0.2

    # Stage 1: Teacher Training (Supervised on Fold 0)
    TEACHER_EPOCHS = 25
    TEACHER_PATIENCE = 6  # Early stopping patience

    # Stage 2: Student Training (Semi-Supervised on Fold 0 + Fold 1 Pseudo-labels)
    STUDENT_EPOCHS = 25
    STUDENT_PATIENCE = 6

    # ==========================================
    # Debugging / Development
    # ==========================================
    # Set to True to run on a small subset of data for pipeline verification
    DEBUG = False
    DEBUG_SUBSET_SIZE = 32
