import os
import torch


class Config:
    """
    Configuration class for the RNA Degradation Prediction experiment.
    Centralizes all constants, paths, and hyperparameters.
    """

    # =========================================================================
    # Reproducibility
    # =========================================================================
    SEED = 42

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_57"
    SUBMISSION_DIR = "./submission"

    # Input Data Paths (Parquet files from metadata generation)
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Sample submission for format reference (optional usage)
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Create necessary directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Data Parameters
    # =========================================================================
    SEQ_LENGTH = 107
    SCORED_LENGTH = 68

    # The specific columns to be predicted and scored
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    NUM_TARGETS = len(TARGET_COLS)

    # Columns to exclude from training (noise reduction)
    EXCLUDE_COLS = ["deg_pH10", "deg_50C"]

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    # Architecture: Vector-Scaled High-Capacity Wide-Stream BiGRU
    HIDDEN_DIM = 512  # Width of the residual stream
    NUM_LAYERS = 6  # Number of residual blocks

    # Embedding Dimensions
    EMBED_DIM_SEQ = 128  # Atomic nucleotide embedding
    EMBED_DIM_LOOP = 64  # Predicted loop type embedding
    EMBED_DIM_PAIR = 64  # Sinusoidal pair distance embedding

    # Regularization
    DROPOUT = 0.2  # Dropout probability in residual branches
    STEM_DROPOUT = 0.0  # No dropout in the initial stem projection

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 32
    EPOCHS = 20

    # Optimizer (AdamW)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4  # Low weight decay to preserve recurrent signals

    # Stability
    GRAD_CLIP_NORM = 1.0  # Critical for Width 512 stability

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2  # Number of dataloader workers
