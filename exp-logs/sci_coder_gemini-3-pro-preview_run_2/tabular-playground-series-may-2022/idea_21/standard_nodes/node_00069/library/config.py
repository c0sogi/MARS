import os


class Config:
    # --------------------------------------------------------------------------
    # Directory and File Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Specific cache directory for this idea iteration
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_21")

    # Raw Data Files
    TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
    TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")
    MODEL_PATH = os.path.join(CACHE_DIR, "best_model.pth")
    PROCESSED_DATA_PATH = os.path.join(CACHE_DIR, "processed_data.npz")

    # --------------------------------------------------------------------------
    # Data Configuration
    # --------------------------------------------------------------------------
    SEED = 42
    TARGET_COL = "target"
    ID_COL = "id"

    # Feature Configuration
    # f_27 is the sequence feature
    SEQUENCE_FEATURE = "f_27"
    SEQUENCE_LENGTH = 10  # f_27 is decomposed into 10 characters

    # --------------------------------------------------------------------------
    # Hyperparameters
    # --------------------------------------------------------------------------
    # Training
    BATCH_SIZE = 1024
    EPOCHS = 40
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler (StepLR)
    SCHEDULER_STEP_SIZE = 10
    SCHEDULER_GAMMA = 0.1

    # Model Architecture
    # Transformer Stream
    EMBED_DIM = 32
    TRANSFORMER_LAYERS = 2
    TRANSFORMER_HEADS = 4
    TRANSFORMER_DROPOUT = 0.1
    TRANSFORMER_ACTIVATION = "gelu"

    # Backbone (Pre-Activation Direct GLU ResFunnel)
    BACKBONE_WIDTHS = [512, 256, 128]
    BACKBONE_DROPOUT = 0.35

    # --------------------------------------------------------------------------
    # Utility
    # --------------------------------------------------------------------------
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)
