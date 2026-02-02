import os
import torch


class Config:
    # --------------------------------------------------------------------------
    # Directory Setup
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching processed data (Idea 11 specific)
    WORKING_DIR = "./working/idea_11"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Submission directory
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # --------------------------------------------------------------------------
    # File Paths
    # --------------------------------------------------------------------------
    # Raw Data
    TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
    TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata (Split definitions)
    TRAIN_META = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Outputs
    CACHE_PATH = os.path.join(WORKING_DIR, "processed_data.npz")
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Hyperparameters
    # --------------------------------------------------------------------------
    # f_00 to f_30 is 31 columns. f_27 is categorical. 30 continuous features.
    NUM_CONTINUOUS_FEATURES = 30

    # f_27 string length is 10
    SEQUENCE_LENGTH = 10

    # A-Z (26 characters) + 1 for padding/unknown = 27
    VOCAB_SIZE = 27

    # --------------------------------------------------------------------------
    # Model Architecture (FiLM-ResFunnel)
    # --------------------------------------------------------------------------
    # Stream 1: Categorical Context
    EMBED_DIM = 32
    TRANSFORMER_LAYERS = 2
    TRANSFORMER_HEADS = 4
    TRANSFORMER_FF_DIM = 128  # 4x embed dim

    # Stream 2: Continuous Signal
    SIGNAL_DIM = 512

    # Backbone: ResFunnel
    # Stages: 512 -> 256 -> 128
    RESFUNNEL_DIMS = [512, 256, 128]

    # Regularization
    DROPOUT_RATE = 0.35

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 512  # Efficient for A100
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2  # Aggressive regularization
    NUM_EPOCHS = 30

    # Scheduler
    SCHEDULER_STEP_SIZE = 10
    SCHEDULER_GAMMA = 0.1

    # --------------------------------------------------------------------------
    # System & Reproducibility
    # --------------------------------------------------------------------------
    RANDOM_SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # For DataLoader
