import os
import torch


class Config:
    """
    Configuration class for the Dual-Stream 1D ResUNet strategy.
    """

    # --- Reproducibility ---
    SEED = 42

    # --- Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_24"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = "./submission"

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Files
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Data Processing Hyperparameters ---

    # Temporal alignment
    SAMPLING_RATE_HZ = 1

    # Feature Engineering
    # We generate two sets of features: Stream A (Context) and Stream B (Precision)

    # Columns to aggregate with statistics (Mean, Std, Min, Max)
    STAT_FEATURES = ["Cn0DbHz", "SvElevationDegrees"]
    STATS_LIST = ["mean", "std", "min", "max"]

    # Columns to aggregate with Mean only
    MEAN_FEATURES = ["RawPseudorangeUncertaintyMeters"]

    # Columns to Count (Satellite Count)
    COUNT_FEATURE = "Svid"

    # High Quality Signal Types for Stream B filtering
    # L5 bands are generally less susceptible to multipath
    HQ_SIGNAL_TYPES = ["GPS_L5", "GAL_E5A", "BDS_B2A", "QZS_J5"]

    # Bitmask for Valid Carrier Phase (Accumulated Delta Range State)
    # Bit 0: Valid, Bit 1: Reset, Bit 2: Cycle Slip
    # We check if ADR_STATE_VALID (1) is set.
    ADR_STATE_VALID_BIT = 1

    # Input Dimensions
    # Calculation: (len(STAT_FEATURES) * 4) + len(MEAN_FEATURES) + 1 (Count)
    # (2 * 4) + 1 + 1 = 10 features per stream
    IN_CHANNELS_A = 10
    IN_CHANNELS_B = 10

    # Sequence Generation
    SEQ_LEN = 128  # Length of time series window input to the model
    TRAIN_STRIDE = 32  # Overlap for training data augmentation
    TEST_STRIDE = 128  # No overlap for inference (sliding window with full step)

    # Target Definition
    # We predict (Delta_East, Delta_North) in meters relative to the WLS baseline
    NUM_CLASSES = 2

    # --- Model Architecture ---
    BASE_FILTERS = 64
    DEPTH = 4  # Number of encoder/decoder blocks
    KERNEL_SIZE = 3  # Convolution kernel size
    DROPOUT = 0.1  # Dropout rate

    # --- Training Hyperparameters ---
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    EPOCHS = 30  # Maximum epochs
    EARLY_STOPPING_PATIENCE = 5

    # Optimizer
    WEIGHT_DECAY = 1e-4
    GRADIENT_CLIP = 1.0

    # Scheduler
    ETA_MIN = 1e-6  # Minimum learning rate for Cosine Annealing

    # Deep Supervision
    # Auxiliary heads attached to decoder layers
    # Loss = Weight_Final * Loss_Final + Weight_Aux * Loss_Aux
    AUX_LOSS_WEIGHT = 0.4
    DECIMATION_FACTOR = 4  # Aux heads trained on ground truth subsampled by this factor

    # --- Hardware ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    @classmethod
    def setup(cls):
        """Ensures necessary directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories on import
Config.setup()
