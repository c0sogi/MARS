import os
import torch


class Config:
    # --------------------------------------------------------------------------
    # General Setup
    # --------------------------------------------------------------------------
    RANDOM_STATE = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------------------------------
    # File Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Specific cache directory for this experiment idea
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_18")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Metadata Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Raw Data Paths
    TRAIN_DATA_PATH = os.path.join(INPUT_DIR, "train.csv")
    TEST_DATA_PATH = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    MODEL_SAVE_PATH = os.path.join(CACHE_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Configuration
    # --------------------------------------------------------------------------
    NUM_CONTINUOUS_FEATURES = 30
    SEQUENCE_LENGTH = 10  # Length of f_27 string
    # Vocabulary size: 26 letters (A-Z) + 1 for potential padding/unknown = 27
    # We map 'A'->1, ..., 'Z'->26. 0 is reserved.
    VOCAB_SIZE = 27

    # --------------------------------------------------------------------------
    # Model Architecture Configuration
    # --------------------------------------------------------------------------
    # Stream 1: Categorical Transformer
    TRANSFORMER_EMBED_DIM = 32
    TRANSFORMER_LAYERS = 2
    TRANSFORMER_HEADS = 4  # Standard choice for dim 32 (32/4 = 8 dim per head)
    TRANSFORMER_DROPOUT = 0.1

    # Stream 2: Continuous
    # Backbone: ResFunnel-GLU
    BACKBONE_STAGES = [512, 256, 128]
    BACKBONE_DROPOUT = 0.35

    # --------------------------------------------------------------------------
    # Training Configuration
    # --------------------------------------------------------------------------
    BATCH_SIZE = 1024
    EPOCHS = 40
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler (StepLR)
    SCHEDULER_STEP_SIZE = 10
    SCHEDULER_GAMMA = 0.1

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 5
