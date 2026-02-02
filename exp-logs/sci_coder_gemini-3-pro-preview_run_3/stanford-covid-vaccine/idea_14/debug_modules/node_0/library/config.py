import os
import torch


class Config:
    """
    Central configuration for the RNA Degradation Prediction task.
    Contains file paths, data specifications, model hyperparameters,
    and training settings.
    """

    # =========================================================================
    # Directories and Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_14"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata File Paths (Parquet format)
    TRAIN_FILE = os.path.join(METADATA_DIR, "train.parquet")
    VAL_FILE = os.path.join(METADATA_DIR, "val.parquet")
    TEST_FILE = os.path.join(METADATA_DIR, "test.parquet")

    # Submission Sample
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    SUBMISSION_FILE = os.path.join(WORKING_DIR, "submission.csv")
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # =========================================================================
    # Data Specifications
    # =========================================================================
    SEQ_LENGTH = 107
    SEQ_SCORED = 68

    # All ground truth columns provided in training data
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Columns specifically used for the competition metric (MCRMSE)
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # Feature Mappings (One-Hot Encoding)
    # Sequence: 4 bases
    TOKEN2INT_SEQ = {x: i for i, x in enumerate("AGUC")}

    # Structure: 3 types (paired open, paired close, unpaired)
    TOKEN2INT_STRUCT = {x: i for i, x in enumerate("().")}

    # Predicted Loop Type: 7 types
    TOKEN2INT_LOOP = {x: i for i, x in enumerate("SMIBHEX")}

    # Input Dimension Calculation:
    # 4 (Seq) + 3 (Struct) + 7 (Loop) = 14
    INPUT_DIM = len(TOKEN2INT_SEQ) + len(TOKEN2INT_STRUCT) + len(TOKEN2INT_LOOP)

    # =========================================================================
    # Model Hyperparameters (Latent Structure-Gated BiGRU)
    # =========================================================================
    # Convolutional Stem
    CONV_FILTERS = 256
    CONV_KERNEL = 3

    # Backbone (BiGRU)
    HIDDEN_DIM = 384
    NUM_LAYERS = 3
    DROPOUT = 0.1

    # Output
    OUTPUT_DIM = len(TARGET_COLS)  # 5

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    BATCH_SIZE = 64
    EPOCHS = 25  # Max epochs, controlled by Early Stopping

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    MAX_GRAD_NORM = 1.0  # Gradient Clipping

    # Scheduler
    T_MAX = EPOCHS  # For CosineAnnealingLR
    ETA_MIN = 1e-6

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2
