import os
import torch


class Config:
    """
    Central configuration for the Audio Tagging task.
    Implements parameters for ConvNeXt-Nano backbone, Dual-Stream Pooling,
    and Multi-Sample Dropout strategy.
    """

    # ==========================================
    # Project & Paths
    # ==========================================
    PROJECT_NAME = "audio_tagging_convnext"
    IDEA_NAME = "idea_6"

    # Input Directories (Read-Only)
    INPUT_ROOT = "./input"
    METADATA_ROOT = "./metadata"

    # Output Directory (Read/Write)
    # Using idea_6 as requested for the specific working directory
    OUTPUT_ROOT = os.path.join("./working", IDEA_NAME)

    # Ensure the output directory exists
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # Metadata File Paths
    TRAIN_CSV = os.path.join(METADATA_ROOT, "train.csv")
    VAL_CSV = os.path.join(METADATA_ROOT, "val.csv")
    TEST_CSV = os.path.join(METADATA_ROOT, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_ROOT, "sample_submission.csv")

    # Checkpoint and Submission Paths
    BEST_MODEL_PATH = os.path.join(OUTPUT_ROOT, "best_model.pth")
    SUBMISSION_PATH = os.path.join(OUTPUT_ROOT, "submission.csv")

    # ==========================================
    # Audio Processing Parameters
    # ==========================================
    SAMPLE_RATE = 32000
    DURATION = 30  # seconds
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512
    FMIN = 20
    FMAX = 16000

    # ==========================================
    # Model Architecture Parameters
    # ==========================================
    BACKBONE = "convnext_nano"
    PRETRAINED = True
    NUM_CLASSES = 80
    IN_CHANNELS = 1  # Modified to accept 1 channel (summed RGB weights)

    # Head Configuration
    POOLING_TYPE = "dual_stream"  # Options: 'avg', 'max', 'dual_stream' (Attn + Max)
    USE_MULTI_SAMPLE_DROPOUT = True
    DROPOUT_RATE = 0.5
    NUM_DROPOUT_SAMPLES = 5

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 32  # Adjusted for A100 (40GB)
    EPOCHS = 25

    # Optimizer & Scheduler
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    MAX_LR = 1e-3  # For OneCycleLR
    PCT_START = 0.3  # For OneCycleLR warm-up

    # Augmentation
    MIXUP_ALPHA = 0.4
    SPECAUG_TIME_MASK = 48
    SPECAUG_FREQ_MASK = 24

    # ==========================================
    # Compute & Hardware
    # ==========================================
    NUM_WORKERS = 12
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Debugging & Development
    # ==========================================
    # Set DEBUG to True to run on a small subset of data for quick testing
    DEBUG = False
    DEBUG_SUBSET_SIZE = 200  # Number of samples to use in debug mode

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print(f"================ {cls.IDEA_NAME} Configuration ================")
        for k, v in cls.__dict__.items():
            if not k.startswith("__") and not callable(v):
                print(f"{k}: {v}")
        print("========================================================")
