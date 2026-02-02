import os
import torch


class Config:
    """
    Central configuration for the Volcano Eruption Prediction task.
    Defines hyperparameters, file paths, and hardware settings for the
    Attention-Pooled Hybrid EfficientNet architecture.
    """

    # --------------------------------------------------------------------------
    # General & Hardware
    # --------------------------------------------------------------------------
    PROJECT_NAME = "volcano_eruption_prediction"
    IDEA_NAME = "idea_4"  # Current experiment identifier
    SEED = 42
    DEBUG = False  # Toggle for debugging with smaller subsets

    # Compute
    NUM_WORKERS = 12  # Utilizing available vCPUs
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------------------------------
    # Directory & File Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = os.path.join("./working", IDEA_NAME)
    SUBMISSION_DIR = "./submission"

    # Input Metadata (Pre-generated)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Output Files
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Files (Parquet for features, NPY for scaler stats)
    # Using Parquet instead of Pickle for safety and efficiency
    TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

    TARGET_SCALER_MEAN = os.path.join(WORKING_DIR, "target_mean.npy")
    TARGET_SCALER_STD = os.path.join(WORKING_DIR, "target_std.npy")
    STATS_SCALER_MEAN = os.path.join(WORKING_DIR, "stats_scaler_mean.npy")
    STATS_SCALER_SCALE = os.path.join(WORKING_DIR, "stats_scaler_scale.npy")

    # --------------------------------------------------------------------------
    # Data Preprocessing Parameters
    # --------------------------------------------------------------------------
    NUM_SENSORS = 10
    SIGNAL_LENGTH = 60001
    SAMPLE_RATE = 100  # Hz (Approximated from 60001 samples in 10 mins)

    # Spectrogram Generation
    # High resolution settings to capture transient tremors
    N_FFT = 1024
    HOP_LENGTH = 256  # Results in time dimension ~235
    N_MELS = 128
    FMIN = 0
    FMAX = 50  # Nyquist frequency for 100Hz sample rate
    TOP_DB = 80.0  # Dynamic range for AmplitudeToDB

    # Statistical Features
    # Features per sensor: Mean, Std, Skew, Kurt, Q05, Q25, Q50, Q75, Q95, NaNs
    NUM_STATS_PER_SENSOR = 10
    NUM_STAT_FEATURES = NUM_SENSORS * NUM_STATS_PER_SENSOR

    # --------------------------------------------------------------------------
    # Model Architecture Parameters
    # --------------------------------------------------------------------------
    BACKBONE = "efficientnet_b0"
    PRETRAINED = True
    IN_CHANNELS = 10  # One channel per sensor
    USE_ATTENTION_POOLING = True  # Gated Attention instead of GAP
    DROPOUT_RATE = 0.2
    HIDDEN_DIM = 128  # Dimension for the fusion layer

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    PATIENCE = 20  # Early stopping patience

    # --------------------------------------------------------------------------
    # Utility Methods
    # --------------------------------------------------------------------------
    @classmethod
    def setup(cls):
        """
        Ensures that working and submission directories exist.
        Should be called at the start of execution.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Automatically setup directories when config is imported
Config.setup()
