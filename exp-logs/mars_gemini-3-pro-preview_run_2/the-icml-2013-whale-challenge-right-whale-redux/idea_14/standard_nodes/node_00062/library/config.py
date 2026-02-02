import os


class Config:
    """
    Configuration class for the Right Whale Detection Task.
    Implements the 'Triple-Architecture Stacked Ensemble' strategy parameters.
    """

    # --- General Configuration ---
    SEED = 42
    NUM_WORKERS = 4

    # --- Directory Paths ---
    INPUT_ROOT = "./input"
    TRAIN_DIR = os.path.join(INPUT_ROOT, "train2")
    TEST_DIR = os.path.join(INPUT_ROOT, "test2")

    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for caching processed data and models
    WORKING_DIR = "./working/idea_15"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Output directory for submission
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Audio Processing Parameters ---
    # High resolution FFT and Hop for transient call detection
    SAMPLE_RATE = 2000
    N_FFT = 1024
    HOP_LENGTH = 64
    N_MELS = 128
    FMIN = 0
    FMAX = None  # Defaults to SR // 2 (1000 Hz)

    # Normalization: False to preserve Pink noise spectral tilt
    MEL_NORMALIZED = False

    # --- Model Architecture ---
    # Ensemble of 2 diverse backbones (Cite Lesson 00037: Remove weaker DenseNet):
    # 1. EfficientNet-B0 (Noisy Student weights)
    # 2. ResNet34 (Standard Residual)
    MODEL_NAMES = ["tf_efficientnet_b0.ns_jft_in1k", "resnet34"]

    NUM_CLASSES = 1
    IN_CHANNELS = 1  # Models adapted to take 1-channel input
    USE_GEM_POOLING = True  # Generalized Mean Pooling

    # --- Training Hyperparameters ---
    BATCH_SIZE = 128
    EPOCHS = 15
    LEARNING_RATE = 1e-3

    # Low weight decay to allow Noisy Student weights to adapt freely
    WEIGHT_DECAY = 1e-4

    NUM_FOLDS = 5
    EARLY_STOPPING_PATIENCE = 5  # Monitor Validation Loss

    # --- Caching Filenames ---
    # Paths for cached numpy arrays to speed up loading
    CACHE_TRAIN_DATA = os.path.join(WORKING_DIR, "train_data.npy")
    CACHE_TRAIN_LABELS = os.path.join(WORKING_DIR, "train_labels.npy")
    CACHE_VAL_DATA = os.path.join(WORKING_DIR, "val_data.npy")
    CACHE_VAL_LABELS = os.path.join(WORKING_DIR, "val_labels.npy")
    CACHE_TEST_DATA = os.path.join(WORKING_DIR, "test_data.npy")
    CACHE_TEST_CLIPS = os.path.join(WORKING_DIR, "test_clips.npy")
