import os
import torch


class Config:
    """
    Configuration for Right Whale Call Detection Pipeline (Idea 7).
    Implements a Heterogeneous Ensemble of EfficientNet-B2 and ResNet-50.
    """

    # --- General ---
    SEED = 42
    DEBUG = False  # Set to True for quick testing with subset of data

    # --- Paths ---
    INPUT_ROOT = "./input"
    TRAIN_DIR = os.path.join(INPUT_ROOT, "train2")
    TEST_DIR = os.path.join(INPUT_ROOT, "test2")

    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Working Directory for Idea 7 (Cache & Models)
    WORKING_DIR = "./working/idea_7"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Submission
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Audio Preprocessing ---
    SAMPLE_RATE = 2000
    N_FFT = 1024  # High freq resolution (Idea)
    HOP_LENGTH = 64  # High temporal resolution (Idea)
    N_MELS = 128
    FMIN = 20  # Low frequency focus for whales
    FMAX = 1000  # Up to Nyquist
    NORMALIZED = False  # Preserve spectral tilt (Idea)

    # --- Input Dimensions ---
    # Resizing spectrograms to standard size for CNN backbones
    IMG_SIZE = (224, 224)
    IN_CHANNELS = 1  # Single channel input (Idea)

    # --- Model Architecture ---
    # Ensemble of Scaled Legacy Architectures (Idea)
    # Using timm naming conventions
    MODEL_ARCHS = ["tf_efficientnet_b2.ns_jft_in1k", "resnet50.a1_in1k"]
    NUM_CLASSES = 1

    # --- Training Hyperparameters ---
    BATCH_SIZE = 128  # Maximize for stability (Idea)
    EPOCHS = 30  # Extended to ensure convergence (Idea)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 8  # Extended early stopping patience (Idea)

    # --- Hardware ---
    NUM_WORKERS = 12  # Utilizing available vCPUs
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Caching Paths ---
    # Used to store processed spectrograms
    TRAIN_DATA_CACHE = os.path.join(WORKING_DIR, "train_data.npy")
    TRAIN_LABELS_CACHE = os.path.join(WORKING_DIR, "train_labels.npy")
    VAL_DATA_CACHE = os.path.join(WORKING_DIR, "val_data.npy")
    VAL_LABELS_CACHE = os.path.join(WORKING_DIR, "val_labels.npy")
    TEST_DATA_CACHE = os.path.join(WORKING_DIR, "test_data.npy")
    TEST_CLIPS_CACHE = os.path.join(WORKING_DIR, "test_clips.npy")
