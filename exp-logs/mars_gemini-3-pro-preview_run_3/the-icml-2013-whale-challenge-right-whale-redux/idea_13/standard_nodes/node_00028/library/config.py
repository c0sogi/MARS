import os
import torch


class Config:
    """
    Configuration for the Right Whale Detection Task.
    Implements the 'Corrected Noisy Student Self-Training Pipeline'.
    """

    # --- Project Structure ---
    PROJECT_NAME = "idea_13"
    WORKING_DIR = os.path.join("./working", PROJECT_NAME)

    # Input Directories (Read-Only)
    INPUT_ROOT = "./input"
    TRAIN_AUDIO_DIR = os.path.join(INPUT_ROOT, "train2")
    TEST_AUDIO_DIR = os.path.join(INPUT_ROOT, "test2")

    # Metadata Paths (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Directories
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINTS_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Audio Processing ---
    # Based on data analysis: SR=2000Hz, Duration=~2s
    SAMPLE_RATE = 2000
    DURATION = 2.0  # seconds
    N_FFT = 1024
    # Hop length of ~10ms. 2000Hz * 0.01s = 20 samples.
    HOP_LENGTH = 20
    N_MELS = 384
    F_MIN = 50
    F_MAX = 1000  # Nyquist frequency at 2000Hz SR

    # --- Model Architecture ---
    # EfficientNetV2-Medium with GeM Pooling
    BACKBONE = "tf_efficientnetv2_m"
    PRETRAINED = True
    NUM_CLASSES = 1
    USE_GEM_POOLING = True

    # Regularization
    DROP_PATH_RATE = 0.2
    DROPOUT_RATE = 0.3

    # --- Training Hyperparameters ---
    SEED = 42
    BATCH_SIZE = 32
    NUM_WORKERS = 4  # Adjust based on vCPU count (12 available)

    # Optimization
    LEARNING_RATE = 1e-3
    MIN_LEARNING_RATE = 1e-6
    WEIGHT_DECAY = 0.01
    EPOCHS = 25
    WARMUP_EPOCHS = 3

    # Augmentation
    MIXUP_ALPHA = 1.0
    SPEC_AUG_TIME_MASK = 30
    SPEC_AUG_FREQ_MASK = 40

    # --- Self-Training / Noisy Student ---
    TEACHER_CONFIDENCE_THRESHOLD = (
        0.0  # Use soft labels, so threshold might not be strict
    )
    STUDENT_NOISE_MULTIPLIER = 1.0  # Factor to scale augmentation intensity if needed

    # --- Debugging / Development ---
    # Flags to control dataset size and runtime for rapid iteration
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 500
    DEBUG_EPOCHS = 2

    # --- Hardware ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def init_directories(cls):
        """Creates necessary working directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINTS_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        print(f"Directories initialized at {cls.WORKING_DIR}")

    @classmethod
    def get_audio_config(cls):
        """Returns a dictionary of audio processing parameters."""
        return {
            "sr": cls.SAMPLE_RATE,
            "n_fft": cls.N_FFT,
            "hop_length": cls.HOP_LENGTH,
            "n_mels": cls.N_MELS,
            "fmin": cls.F_MIN,
            "fmax": cls.F_MAX,
            "duration": cls.DURATION,
        }
