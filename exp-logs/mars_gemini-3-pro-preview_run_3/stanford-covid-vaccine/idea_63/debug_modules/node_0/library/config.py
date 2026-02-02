import os
import torch


class Config:
    """
    Configuration class for the High-Capacity Stabilized Decoupled Bias-Refined BiGRU (HC-SDBR-BiGRU) strategy.
    Centralizes all file paths, model hyperparameters, and training settings.
    """

    # ==========================================
    # File Paths and Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_63"
    SUBMISSION_DIR = "./submission"

    # Input Files
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.parquet")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.parquet")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.parquet")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Ensure necessary write directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Compute and Reproducibility
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Optimized for 12 vCPUs

    # ==========================================
    # Data Specifications
    # ==========================================
    SEQ_LEN = 107
    SEQ_SCORED = 68

    # Input Features: 4 (Nucleotide) + 3 (Structure) + 7 (Loop Type)
    INPUT_CHANNELS = 14

    # Targets
    NUM_TARGETS = 5
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # Only these columns are used for the final metric calculation during validation
    SCORED_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # ==========================================
    # Model Hyperparameters (HC-SDBR-BiGRU)
    # ==========================================
    # Convolutional Stem
    CNN_FILTERS = 256
    CNN_KERNEL_SIZE = 3

    # Backbone Capacity
    # Hidden dimension per direction. Total BiGRU output will be HIDDEN_DIM * 2 = 768.
    HIDDEN_DIM = 384
    NUM_LAYERS = 4
    DROPOUT = 0.1

    # Architectural Flags for Stability and Performance
    USE_POST_NORM = True  # Stabilizes deep 4-layer stack
    USE_INTERNAL_GATE_NORM = True  # Prevents saturation in interaction gates
    BIAS_REFINED_INTERACTION = True  # Allows unpaired bases to self-refine via bias

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 25

    # Gradient Clipping is mandatory for this hybrid architecture
    MAX_GRAD_NORM = 1.0

    # Scheduler settings (Cosine Annealing)
    T_MAX = EPOCHS
    ETA_MIN = 1e-6

    # ==========================================
    # Debugging and Development
    # ==========================================
    # Set to True to run on a small subset of data for pipeline verification
    DEBUG = False
    DEBUG_SUBSET_SIZE = 100
