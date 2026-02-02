import os
import torch


class Config:
    """
    Global configuration for the Deep Residual MLP pipeline.
    """

    # --------------------------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------------------------
    RANDOM_SEED = 42

    # --------------------------------------------------------------------------
    # File Paths & Directories
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_3"
    SUBMISSION_DIR = "./submission"

    # Ensure necessary output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Raw Data
    TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
    TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata (Stratified Splits)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Processed Data Cache
    PROCESSED_DATA_PATH = os.path.join(WORKING_DIR, "processed_data.npz")

    # Model Artifacts
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Processing Parameters
    # --------------------------------------------------------------------------
    # Number of continuous features in the dataset (f_00 to f_26, f_28 to f_30)
    NUM_CONTINUOUS_FEATURES = 30

    # Length of the string feature 'f_27'
    F_27_SEQ_LENGTH = 10

    # Vocabulary size for character embedding (A-Z + padding)
    # 26 letters + 1 padding = 27. Setting to 35 for safety.
    VOCAB_SIZE = 35

    # --------------------------------------------------------------------------
    # Model Architecture (Deep ResMLP)
    # --------------------------------------------------------------------------
    # Dimension for character embeddings
    EMBEDDING_DIM = 32

    # Main hidden dimension for the residual backbone
    HIDDEN_DIM = 512

    # Number of residual blocks (Depth)
    NUM_RES_BLOCKS = 6

    # Dropout rate for regularization within blocks
    DROPOUT_RATE = 0.1

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    # Large batch size for tabular data on GPU
    BATCH_SIZE = 2048

    # Learning rate
    LEARNING_RATE = 1e-3

    # Weight decay for AdamW (Aggressive regularization as per strategy)
    WEIGHT_DECAY = 1e-2

    # Training duration (Extended training as per strategy)
    EPOCHS = 35

    # Hardware acceleration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Data loading workers
    NUM_WORKERS = 4

    # --------------------------------------------------------------------------
    # Debugging / Development
    # --------------------------------------------------------------------------
    # Set to True to use a small subset of data for rapid prototyping
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 5000
