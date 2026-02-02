import os
import torch


class Config:
    # --------------------------------------------------------------------------
    # System & Reproducibility
    # --------------------------------------------------------------------------
    PROJECT_NAME = "ResFunnel_GLU_Scaled"
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Adjust based on available vCPUs (12 available)

    # --------------------------------------------------------------------------
    # File Paths & Directories
    # --------------------------------------------------------------------------
    # Base Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_7"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Raw Data
    TRAIN_DATA_PATH = os.path.join(INPUT_DIR, "train.csv")
    TEST_DATA_PATH = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata (Stratified Splits)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Outputs
    PROCESSED_DATA_PATH = os.path.join(WORKING_DIR, "processed_data.npz")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Processing Hyperparameters
    # --------------------------------------------------------------------------
    # f_27 is a string of length 10. We decompose it into 10 integer tokens.
    F27_SEQ_LEN = 10
    # Vocabulary size for characters (A-Z).
    # 26 letters + 1 for potential padding/unknown (though data is clean) -> 27
    VOCAB_SIZE = 27
    # Number of continuous features (f_00 to f_30, excluding f_27)
    NUM_CONT_FEATURES = 30

    # --------------------------------------------------------------------------
    # Model Architecture: ResFunnel-GLU
    # --------------------------------------------------------------------------
    # Dimension for character embeddings
    EMBED_DIM = 32

    # Width of the initial projection before the funnel begins
    INIT_WIDTH = 1024

    # Widths for the progressive compression stages
    # Stage 1: 1024, Stage 2: 512, Stage 3: 256
    STAGES = [1024, 512, 256]

    # Aggressive dropout to prevent overfitting in deep residual networks (Cite Lesson 16)
    DROPOUT_RATE = 0.40

    # --------------------------------------------------------------------------
    # Training & Optimization
    # --------------------------------------------------------------------------
    # Large batch size for stable gradients and speed
    BATCH_SIZE = 2048

    # Max epochs (Early stopping will likely trigger sooner)
    MAX_EPOCHS = 60

    # Optimizer settings
    LEARNING_RATE = 1e-3
    # High weight decay as requested for regularization
    WEIGHT_DECAY = 1e-2

    # Early Stopping (Cite Lesson 3: Decoupling Loss Convergence from Ranking Improvement)
    PATIENCE = 15

    # --------------------------------------------------------------------------
    # Debugging
    # --------------------------------------------------------------------------
    # If True, runs on a small subset of data for rapid prototyping
    DEBUG = False
    DEBUG_SAMPLES = 10000
