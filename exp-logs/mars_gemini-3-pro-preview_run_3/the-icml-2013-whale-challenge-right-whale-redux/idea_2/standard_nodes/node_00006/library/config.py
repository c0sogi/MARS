import os
import torch
import numpy as np
import random


class Config:
    """
    Configuration for the Right Whale Detection Task using SE-ResNet.
    """

    # --- File Paths ---
    INPUT_DIR = "./input"
    TRAIN_AUDIO_DIR = os.path.join(INPUT_DIR, "train2")
    TEST_AUDIO_DIR = os.path.join(INPUT_DIR, "test2")

    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for caching processed data (Idea 2)
    WORKING_DIR = "./working/idea_2"
    CACHE_DIR = WORKING_DIR

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Audio Preprocessing Parameters ---
    SAMPLE_RATE = 2000
    DURATION = 2.0  # seconds (based on max duration in analysis)
    N_SAMPLES = int(SAMPLE_RATE * DURATION)  # 4000 samples

    # Log-Mel Spectrogram
    # Target shape: approx 128x128 for ResNet compatibility
    N_FFT = 256
    HOP_LENGTH = 32  # 4000 / 32 = 125 time frames
    N_MELS = 128
    FMIN = 20
    FMAX = 1000  # Nyquist frequency at 2000Hz SR

    # --- Model Parameters ---
    MODEL_NAME = "se_resnet"
    IN_CHANNELS = 1  # Mono audio converted to 1-channel spectrogram
    BASE_CHANNELS = 32  # Base filter count for the network
    NUM_CLASSES = 1  # Binary classification

    # --- Training Hyperparameters ---
    SEED = 42
    BATCH_SIZE = 64
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 10  # For Early Stopping

    # Class Imbalance Handling
    # Positive class is ~10% of data. Weight = Neg_Count / Pos_Count ≈ 9.0
    POS_WEIGHT = 9.0

    # --- Augmentation ---
    MIXUP_ALPHA = 0.4
    SPEC_AUG_FREQ_MASK = 15
    SPEC_AUG_TIME_MASK = 20

    # --- Hardware ---
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Initializes the environment by creating necessary directories
        and setting random seeds for reproducibility.
        """
        # Ensure writable directories exist
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set seeds
        cls.set_seed(cls.SEED)

    @staticmethod
    def set_seed(seed):
        """
        Sets the seed for random number generators in Python, NumPy, and PyTorch.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)
