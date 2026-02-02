import os
import torch


class Config:
    """
    Configuration class for Right Whale Detection task.
    Centralizes file paths, audio processing parameters, model hyperparameters,
    and training settings.
    """

    # ==========================================
    # File Paths and Directories
    # ==========================================
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"

    # Metadata CSVs
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Working Directory for Caching and Checkpoints
    # Using 'idea_1' as per the baseline approach identifier
    WORKING_DIR = "./working/idea_1"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Submission Directory
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Files for Preprocessed Data
    # We use .npy format for efficient storage of spectrogram tensors
    CACHE_TRAIN_DATA = os.path.join(WORKING_DIR, "train_data.npy")
    CACHE_TRAIN_LABELS = os.path.join(WORKING_DIR, "train_labels.npy")
    CACHE_VAL_DATA = os.path.join(WORKING_DIR, "val_data.npy")
    CACHE_VAL_LABELS = os.path.join(WORKING_DIR, "val_labels.npy")
    CACHE_TEST_DATA = os.path.join(WORKING_DIR, "test_data.npy")
    CACHE_TEST_IDS = os.path.join(WORKING_DIR, "test_ids.npy")

    # Model Checkpoint Path
    MODEL_CHECKPOINT = os.path.join(WORKING_DIR, "best_model.pth")

    # ==========================================
    # Audio Processing Parameters
    # ==========================================
    # Based on dataset analysis: Sample Rate is 2kHz, Max Duration is 2.0s
    SAMPLE_RATE = 2000
    DURATION = 2.0  # seconds
    TARGET_LENGTH = int(SAMPLE_RATE * DURATION)  # 4000 samples

    # Mel Spectrogram Parameters
    # Right whale calls are low frequency (up-calls ~50-250Hz)
    # High temporal resolution is beneficial.
    N_MELS = 64
    N_FFT = 256  # Window size for FFT
    WIN_LENGTH = 50  # 25ms * 2000Hz = 50 samples
    HOP_LENGTH = 20  # 10ms * 2000Hz = 20 samples
    F_MIN = 20.0  # Minimum frequency
    F_MAX = 1000.0  # Nyquist frequency

    # ==========================================
    # Model Architecture Parameters
    # ==========================================
    # Bi-directional Recurrent Spectral Network
    INPUT_SIZE = N_MELS  # Input features per time step
    HIDDEN_SIZE = 64  # Hidden state size of GRU
    NUM_LAYERS = 2  # Number of stacked GRU layers
    DROPOUT = 0.3  # Dropout probability

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 30
    PATIENCE = 7  # Early stopping patience

    # Class Imbalance Handling
    # Positive class is ~10% of data.
    # pos_weight = (num_neg / num_pos) ~= 9.0
    POS_WEIGHT = 9.0

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # For data loading
