import os
import torch


class Config:
    # ==========================================
    # Project & Experiment Identification
    # ==========================================
    PROJECT_NAME = "RightWhaleDetection"
    EXPERIMENT_NAME = "idea_8"
    SEED = 42

    # ==========================================
    # Directory Paths
    # ==========================================
    # Input directories (Read-Only)
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train2")
    TEST_DIR = os.path.join(INPUT_DIR, "test2")

    # Metadata paths
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sampleSubmission.csv")

    # Working directory (Read/Write)
    WORKING_DIR = os.path.join("./working", EXPERIMENT_NAME)

    # Output paths
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")
    MODEL_A_PATH = os.path.join(WORKING_DIR, "model_a_best.pth")
    MODEL_B_PATH = os.path.join(WORKING_DIR, "model_b_best.pth")

    # ==========================================
    # Audio Processing Parameters
    # ==========================================
    # High-fidelity settings as per strategy
    SAMPLE_RATE = 2000  # Native sample rate of the dataset
    DURATION = 2.0  # Max duration in seconds
    N_FFT = 1024  # High frequency resolution
    HOP_LENGTH = 64  # High temporal resolution
    N_MELS = 128  # Number of Mel bands
    FMIN = 0
    FMAX = None  # Defaults to SR // 2

    # Preprocessing Logic
    # We do NOT resize to 224x224. We use native resolution.
    # Approx shape: (128 mels, ~63 time steps)
    RESIZE_IMG = False

    # Normalization
    # Instance Normalization: (x - mean) / std per clip
    NORMALIZE_INSTANCE = True

    # ==========================================
    # Model Architecture
    # ==========================================
    # Ensemble Members
    MODEL_A_NAME = "tf_efficientnet_b0.ns_jft_in1k"  # Noisy Student weights
    MODEL_B_NAME = "densenet121"  # DenseNet architecture

    PRETRAINED = True
    IN_CHANNELS = 1  # Modified to accept 1-channel spectrograms
    NUM_CLASSES = 1
    USE_GEM_POOLING = True  # Generalized Mean Pooling

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 128  # Maximized for stability
    NUM_EPOCHS = 25  # Sufficient for convergence
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    PATIENCE = 7  # Early stopping patience

    # Optimization
    OPTIMIZER = "AdamW"
    SCHEDULER = "CosineAnnealingLR"
    MIN_LR = 1e-6

    # ==========================================
    # Hardware & Computation
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Number of dataloader workers

    # ==========================================
    # Augmentation
    # ==========================================
    SPEC_AUGMENT = True
    FREQ_MASK_PARAM = 15
    TIME_MASK_PARAM = 15

    # ==========================================
    # Utility Methods
    # ==========================================
    @classmethod
    def setup(cls):
        """Ensures working directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        print(f"Configuration initialized. Working directory: {cls.WORKING_DIR}")
        print(f"Device: {cls.DEVICE}")


# Initialize setup immediately when imported to ensure directories exist
Config.setup()
