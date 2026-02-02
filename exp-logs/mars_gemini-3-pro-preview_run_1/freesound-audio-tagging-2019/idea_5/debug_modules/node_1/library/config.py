import os
import torch


class Config:
    """
    Configuration class for Audio Tagging Task.
    Centralizes all hyperparameters, file paths, and model settings.
    """

    # ==========================================
    # General Settings
    # ==========================================
    PROJECT_NAME = "idea_5"
    SEED = 42
    DEBUG = False  # Set to True for quick debugging runs
    DEBUG_SUBSET_SIZE = 200  # Number of samples to use in debug mode

    # ==========================================
    # File Paths
    # ==========================================
    # Input Directories (Read-Only)
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output Directories (Write Allowed)
    # The working directory for this specific idea/experiment
    OUTPUT_DIR = os.path.join("./working", PROJECT_NAME)

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Distillation / Caching Paths
    # Path to save the Teacher model weights
    TEACHER_MODEL_PATH = os.path.join(OUTPUT_DIR, "teacher_best.pth")
    # Path to save the Student model weights
    STUDENT_MODEL_PATH = os.path.join(OUTPUT_DIR, "student_best.pth")
    # Path to store Teacher predictions on Noisy data (Soft Labels)
    TEACHER_PREDS_NPY = os.path.join(OUTPUT_DIR, "teacher_preds_noisy.npy")

    # ==========================================
    # Audio Preprocessing Parameters
    # ==========================================
    SR = 32000  # Sampling Rate (Resample to this)
    DURATION = 5  # Duration in seconds for training crops
    N_MELS = 128  # Number of Mel bands
    N_FFT = 1024  # FFT window size
    HOP_LENGTH = 320  # Hop length for STFT
    FMIN = 20  # Min frequency
    FMAX = 16000  # Max frequency

    # ==========================================
    # Model Architecture
    # ==========================================
    BACKBONE = "efficientnet_b3"
    PRETRAINED = True  # Use ImageNet weights
    IN_CHANNELS = 1  # Input spectrogram channels
    INPUT_REPETITION = True  # Repeat 1ch input to 3ch for backbone compatibility
    NUM_CLASSES = 80

    # Head & Aggregation
    POOLING = "attention"  # Options: 'attention', 'avg', 'max'
    USE_MULTI_SAMPLE_DROPOUT = True
    DROPOUT_RATE = 0.5  # Dropout rate (if not using Multi-Sample)

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 48  # Adjust based on GPU memory (A100 40GB allows larger batches)
    EPOCHS = 28  # Long schedule for Mixup convergence

    # Optimizer (AdamW)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    MIN_LR = 1e-6  # For Cosine Annealing

    # ==========================================
    # Augmentation Strategies
    # ==========================================
    # Mixup
    MIXUP = True
    MIXUP_ALPHA = 0.4
    MIXUP_PROB = 1.0  # Apply to 100% of batches

    # SpecAugment
    SPECAUGMENT = True
    TIME_MASK_PARAM = 80
    FREQ_MASK_PARAM = 24

    # ==========================================
    # System / Hardware
    # ==========================================
    NUM_WORKERS = 8  # Number of DataLoader workers
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Ensures all necessary output directories exist.
        Must be called at the start of the pipeline.
        """
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        print(f"Directories initialized: {cls.OUTPUT_DIR}, {cls.SUBMISSION_DIR}")

    @classmethod
    def get_transforms_config(cls):
        """Returns a dictionary of transform configurations."""
        return {
            "sr": cls.SR,
            "n_mels": cls.N_MELS,
            "n_fft": cls.N_FFT,
            "hop_length": cls.HOP_LENGTH,
            "fmin": cls.FMIN,
            "fmax": cls.FMAX,
        }
