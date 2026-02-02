import os
import torch


class Config:
    # ==========================================
    #                 Paths
    # ==========================================
    INPUT_ROOT = "./input"
    TRAIN_DIR = os.path.join(INPUT_ROOT, "train2")
    TEST_DIR = os.path.join(INPUT_ROOT, "test2")

    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for caching intermediate data and model checkpoints
    WORKING_DIR = "./working/idea_12"

    # Submission directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    #            Audio / Spectrogram
    # ==========================================
    # "Golden Recipe" Signal Processing Parameters
    SAMPLE_RATE = 2000
    N_FFT = 1024
    HOP_LENGTH = 64
    N_MELS = 128
    FMIN = 0
    FMAX = None  # Defaults to Nyquist (SR/2)

    # Critical: Disable normalization to preserve Pink noise spectral tilt
    MEL_NORMALIZED = False

    # Input Dimensions
    # Analysis showed max duration is 2.0s.
    # We pad all clips to 2.0s (4000 samples) to allow batching at "Native Resolution".
    # We do NOT resize to square (e.g., 224x224).
    MAX_DURATION = 2.0
    NUM_SAMPLES = int(MAX_DURATION * SAMPLE_RATE)  # 4000 samples

    # ==========================================
    #               Model
    # ==========================================
    # Level 0 Base Learners: Heterogeneous Legacy Encoders
    # Using standard 'tf_efficientnet_b0' as supervised weights transfer better to audio (Cite solution_lesson_node_00044)
    MODELS = ["tf_efficientnet_b0", "resnet34"]

    IN_CHANNELS = 1
    NUM_CLASSES = 1
    USE_PRETRAINED = True
    POOLING = "gem"  # Generalized Mean Pooling

    # ==========================================
    #              Training
    # ==========================================
    SEED = 42
    N_FOLDS = 5
    BATCH_SIZE = 128
    NUM_WORKERS = 4  # Optimized for 12 vCPUs

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Training Loop
    EPOCHS = 20
    EARLY_STOPPING_PATIENCE = 6

    # Scheduler: Cosine Annealing
    T_MAX = 20
    ETA_MIN = 1e-6

    # Compute
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """Ensures necessary directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
