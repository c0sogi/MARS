import os
import torch


class Config:
    # ==========================================
    # Paths
    # ==========================================
    INPUT_DIR = "./input"
    TRAIN_AUDIO_DIR = os.path.join(INPUT_DIR, "train", "audio")
    TEST_AUDIO_DIR = os.path.join(INPUT_DIR, "test", "audio")
    METADATA_DIR = "./metadata"

    # Experiment specific working directory
    WORKING_DIR = "./working/idea_6"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Audio Parameters
    # ==========================================
    SAMPLE_RATE = 16000
    DURATION = 1.0  # seconds
    NUM_SAMPLES = int(SAMPLE_RATE * DURATION)

    # ==========================================
    # Multi-Resolution Spectrogram Parameters
    # ==========================================
    # We use 3 channels (RGB) corresponding to different STFT resolutions
    N_MELS = 64
    F_MIN = 20
    F_MAX = 8000

    # Shared hop length ensures time dimensions align across channels
    # 10ms hop
    HOP_LENGTH = 160

    # Channel 1: Short window (High Temporal Resolution) -> 20ms
    # Channel 2: Medium window (Balanced) -> 40ms
    # Channel 3: Long window (High Frequency Resolution) -> 60ms
    WIN_LENGTHS = [320, 640, 960]
    N_FFTS = [512, 1024, 2048]

    # ==========================================
    # Model Parameters
    # ==========================================
    NUM_CLASSES = 12
    # SK-ResNet34 parameters handled by model definition, but we can define neck params here
    GRU_HIDDEN_SIZE = 128
    GRU_LAYERS = 1
    DROPOUT = 0.3

    # ==========================================
    # Training Parameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 20
    EARLY_STOPPING_PATIENCE = 6
    NUM_WORKERS = 4

    # Debug / Subsampling
    DEBUG = False
    MAX_TRAIN_SAMPLES = None  # Set to integer to limit dataset size for debugging

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # Labels
    # ==========================================
    LABELS = [
        "yes",
        "no",
        "up",
        "down",
        "left",
        "right",
        "on",
        "off",
        "stop",
        "go",
        "silence",
        "unknown",
    ]
    LABEL2ID = {l: i for i, l in enumerate(LABELS)}
    ID2LABEL = {i: l for i, l in enumerate(LABELS)}

    @classmethod
    def setup(cls):
        """Creates necessary directories for the experiment."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set deterministic behavior
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
