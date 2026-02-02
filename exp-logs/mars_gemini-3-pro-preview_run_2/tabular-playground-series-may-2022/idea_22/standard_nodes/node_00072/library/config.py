import os
import torch


class Config:
    # --------------------------------------------------------------------------
    # File Paths & Directories
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Specific cache directory for this idea (Idea 22)
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_22")

    # Raw Data Paths
    TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
    TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Paths
    TRAIN_META = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Paths
    PROCESSED_DATA_PATH = os.path.join(CACHE_DIR, "processed_data.npz")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Parameters
    # --------------------------------------------------------------------------
    # f_00 to f_30 are continuous features
    NUM_CONT_FEATURES = 30  # f_00 ... f_29? No, f_00..f_29 is 30 features. f_30 exists?
    # Checking EDA: "Numerical Columns: 30". "f_30" is listed in importance.
    # Wait, EDA says "Numerical Columns: 30".
    # Let's assume standard range f_00 to f_30 might imply 31 features if inclusive,
    # or f_00-f_29 if 0-indexed count.
    # EDA output shows "f_30" in feature importance.
    # EDA output shows "Numerical Columns: 30".
    # If f_30 is present, and count is 30, maybe one is missing?
    # Actually, usually in this dataset f_27 is the string.
    # f_00..f_26 (27 feats) + f_28..f_30 (3 feats) = 30 continuous features.
    # We will detect this dynamically in dataset class, but setting a default here.

    # Sequence Feature (f_27)
    SEQ_LEN = 10  # Length of the string in f_27
    VOCAB_SIZE = 30  # A-Z (26) + Padding/Unknown. Sufficient buffer.

    # --------------------------------------------------------------------------
    # Model Architecture
    # --------------------------------------------------------------------------
    # Transformer Stream (Stream 1)
    EMBED_DIM = 32
    TRANSFORMER_LAYERS = 2
    TRANSFORMER_HEADS = 4
    TRANSFORMER_ACTIVATION = "gelu"
    DROPOUT_TRANSFORMER = 0.1

    # Backbone (Stream 2 + Fusion)
    # Stages: 512 -> 256 -> 128
    BACKBONE_STAGES = [512, 256, 128]
    # "Sustained Depth": 3 blocks per stage
    BLOCKS_PER_STAGE = 3
    DROPOUT_BACKBONE = 0.35

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    RANDOM_STATE = 42
    BATCH_SIZE = 1024
    EPOCHS = 40

    # Optimizer
    LEARNING_RATE = 1e-3  # Standard starting point for AdamW
    WEIGHT_DECAY = 1e-2

    # Scheduler (Aggressive Step Decay)
    SCHEDULER_STEP_SIZE = 10
    SCHEDULER_GAMMA = 0.1

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Adjust based on vCPUs (12 available)

    @classmethod
    def setup(cls):
        """Ensures necessary directories exist."""
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.WORKING_DIR, exist_ok=True)


# Initialize directories immediately upon import
Config.setup()
