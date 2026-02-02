import os
import torch


class Config:
    # ==========================================
    # Project & Experiment Setup
    # ==========================================
    PROJECT_NAME = "RNA_Degradation_Prediction"
    IDEA_NAME = "idea_82"  # Unique identifier for this run
    SEED = 42

    # ==========================================
    # File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = os.path.join("./working", IDEA_NAME)

    # Metadata Paths (Parquet files)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Submission Sample
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Outputs
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # ==========================================
    # Data Specifications
    # ==========================================
    SEQ_LENGTH = 107
    SEQ_SCORED = 68

    # Input Features:
    # 4 (A,G,C,U) + 3 (.,(,)) + 7 (S,M,I,B,H,E,X) = 14 channels
    INPUT_CHANNELS = 14

    # Targets
    NUM_TARGETS = 5
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # Only these 3 are used for the competition metric
    SCORED_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # ==========================================
    # Model Hyperparameters
    # Strategy: High-Capacity GLU-Refined Decoupled BiGRU
    # ==========================================
    HIDDEN_DIM = 768  # High capacity: 384 per direction * 2
    NUM_LAYERS = 4  # Deep hierarchy
    DROPOUT = 0.1  # Conservative regularization

    # Convolutional Stem
    STEM_FILTERS = 256
    STEM_KERNEL_SIZE = 3

    # Interaction Module
    # Wide gate dimension for MLP
    GATE_HIDDEN_DIM = 768

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 50
    PATIENCE = 10  # Early stopping patience

    # Stability
    MAX_GRAD_NORM = 1.0  # Gradient clipping

    # Scheduler
    T_MAX = 50  # For CosineAnnealingLR
    ETA_MIN = 1e-6

    # ==========================================
    # Hardware & Execution
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2

    # Debugging / Development
    # Set DEBUG to True to run on a small subset of data
    DEBUG = False
    DEBUG_SAMPLES = 100

    @classmethod
    def setup(cls):
        """
        Initializes the working directory.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        print(f"Config configured for {cls.DEVICE}. Working dir: {cls.WORKING_DIR}")
