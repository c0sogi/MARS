import os
import torch


class Config:
    # ==========================================
    # Paths and Directories
    # ==========================================
    INPUT_ROOT = "./input"
    TRAIN_DIR = os.path.join(INPUT_ROOT, "train2")
    TEST_DIR = os.path.join(INPUT_ROOT, "test2")

    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    WORKING_DIR = "./working/idea_15"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Create necessary writable directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Signal Processing
    # ==========================================
    SR = 2000  # Sampling rate
    DURATION = 2.0  # Clip duration in seconds
    N_MELS = 384  # Frequency resolution for V2-Medium
    N_FFT = 1024  # Large window for frequency resolution
    HOP_LENGTH = 20  # ~10ms hop at 2000Hz
    FMIN = 0
    FMAX = None  # Nyquist

    # ==========================================
    # Model Architecture
    # ==========================================
    MODEL_NAME = "tf_efficientnetv2_m"  # timm backbone
    POOLING = "gem"  # Generalized Mean Pooling
    IN_CHANNELS = 1
    NUM_CLASSES = 1
    PRETRAINED = True

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 64
    NUM_WORKERS = 4
    EPOCHS = 15  # Total training epochs per stage
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # ==========================================
    # Augmentation (Mixup & SpecAugment)
    # ==========================================
    MIXUP_ALPHA = 0.4  # Calibrated mixup intensity
    FREQ_MASK_PARAM = 20
    TIME_MASK_PARAM = 20

    # ==========================================
    # Differential Structural Regularization
    # ==========================================
    # Stage 1: Teacher (Conservative)
    TEACHER_DROP_PATH_RATE = 0.1
    TEACHER_DROPOUT_RATE = 0.2

    # Stage 2: Student (Noisy/Robust)
    STUDENT_DROP_PATH_RATE = 0.3
    STUDENT_DROPOUT_RATE = 0.5

    # ==========================================
    # Execution & Debugging
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debugging / Development
    DEBUG = False  # Set to True to run on a subset
    DEBUG_SAMPLES = 500  # Number of samples to use in debug mode
