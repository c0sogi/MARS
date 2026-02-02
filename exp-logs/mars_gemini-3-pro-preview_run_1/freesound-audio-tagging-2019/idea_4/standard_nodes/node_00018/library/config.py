import os
import torch


class Config:
    """
    Configuration class for Audio Tagging Task.
    Centralizes all hyperparameters for data processing, model architecture, and training.
    """

    # =======================
    # Project & Paths
    # =======================
    PROJECT_NAME = "idea_4"
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"

    # Output directories
    # The working directory for checkpoints and logs
    OUTPUT_DIR = os.path.join("./working", PROJECT_NAME)

    # The directory for the final submission file
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata file paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Create directories if they don't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =======================
    # Audio Processing
    # =======================
    # Using 32kHz to balance quality and computational efficiency
    SR = 32000
    N_FFT = 1024
    HOP_LENGTH = 320
    N_MELS = 128
    FMIN = 20
    FMAX = 16000  # Nyquist frequency for 32kHz

    # =======================
    # Data Loading & Augmentation
    # =======================
    # Training uses random crops, Validation/Test uses full length
    TRAIN_DURATION = 5.0  # seconds
    VAL_DURATION = None  # Process full audio clip
    TEST_DURATION = None  # Process full audio clip

    BATCH_SIZE = 64
    NUM_WORKERS = 4

    # Augmentation Hyperparameters
    # Continuous Mixup applied to 100% of batches
    MIXUP_ALPHA = 0.4
    MIXUP_PROB = 1.0

    # SpecAugment
    SPEC_AUG_TIME_MASK = 30
    SPEC_AUG_FREQ_MASK = 20

    # =======================
    # Model Architecture
    # =======================
    MODEL_NAME = "efficientnet_b3"
    PRETRAINED = True
    NUM_CLASSES = 80
    IN_CHANNELS = 1

    # Structural adaptations
    # Repeat mono input 3 times to match ImageNet pretrained weights
    USE_INPUT_REPETITION = True
    # Use learnable Batch Normalization before backbone
    USE_LEARNABLE_BN = True
    # Aggregation mechanism
    POOLING_TYPE = "attention"

    # =======================
    # Training Hyperparameters
    # =======================
    SEED = 42
    # Extended training schedule to accommodate heavy Mixup regularization
    EPOCHS = 28

    # Optimizer (AdamW)
    LR = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler (Cosine Annealing)
    T_MAX = 28  # Should match EPOCHS
    MIN_LR = 1e-6

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 7

    # =======================
    # System & Debugging
    # =======================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debug mode to run on a smaller subset of data
    DEBUG = False
    DEBUG_SUBSET_SIZE = 100

    @classmethod
    def print_config(cls):
        """Prints the configuration."""
        print(f"=== Configuration: {cls.PROJECT_NAME} ===")
        print(f"Device: {cls.DEVICE}")
        print(f"Model: {cls.MODEL_NAME} (Pretrained: {cls.PRETRAINED})")
        print(f"Input: {cls.SR}Hz, {cls.N_MELS} Mels, {cls.TRAIN_DURATION}s Crops")
        print(f"Mixup: Alpha={cls.MIXUP_ALPHA}, Prob={cls.MIXUP_PROB}")
        print(f"Training: {cls.EPOCHS} Epochs, BS={cls.BATCH_SIZE}, LR={cls.LR}")
        print(f"Output: {cls.OUTPUT_DIR}")
        print("==========================================")
