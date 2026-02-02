import os
import torch


class Config:
    """
    Configuration for the Full-Scale Multi-Resolution Dense-Hybrid Network (FMDH-Net).
    Defines paths, hyperparameters, model architecture settings, and feature names.
    """

    # =========================================================================
    # 1. File Paths & Directories
    # =========================================================================
    # Input Metadata (Pre-split)
    METADATA_DIR = "./metadata"
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output & Cache
    WORKING_DIR = "./working/idea_15"
    CACHE_DIR = WORKING_DIR
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # 2. Reproducibility & Hardware
    # =========================================================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # For DataLoader

    # =========================================================================
    # 3. Training Hyperparameters (Critical Mass Regime)
    # =========================================================================
    # Strict adherence to Lesson 54: Full dataset, small batch, long training
    BATCH_SIZE = 128
    EPOCHS = 80

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_PATIENCE = 15

    # Debugging / Development
    # Set DEBUG = True to run on a small subset for rapid code verification
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 5000  # Number of breaths to use in debug mode

    # =========================================================================
    # 4. Model Architecture (FMDH-Net)
    # =========================================================================
    # Branch 1: Multi-Resolution Dense CNN (Resistive Stream)
    CNN_FILTERS = 64
    CNN_KERNELS = [3, 7, 11]  # Multi-scale local feature extraction
    CNN_DROPOUT = 0.1

    # Branch 2: High-Capacity Bidirectional LSTM (Elastic Stream)
    LSTM_HIDDEN_SIZE = 512
    LSTM_LAYERS = 3
    LSTM_BIDIRECTIONAL = True
    LSTM_DROPOUT = 0.1

    # Fusion Head
    FC_DROPOUT = 0.1

    # =========================================================================
    # 5. Data Pipeline & Feature Engineering
    # =========================================================================
    # Sequence Information
    SEQ_LEN = 80  # Fixed breath length

    # Column Definitions
    ID_COL = "id"
    BREATH_ID_COL = "breath_id"
    TARGET_COL = "pressure"

    # Raw Input Columns
    RAW_CONT_COLS = ["time_step", "u_in"]
    RAW_CAT_COLS = ["u_out", "R", "C"]

    # Feature Names (Must match the output of the feature engineering pipeline)
    # 1. Dynamic State
    FEAT_DYNAMIC = [
        "time_step",
        "u_in",
        "u_out",
        "area",  # Integral of u_in
        "u_in_diff",  # Derivative of u_in
    ]

    # 2. Lookahead Features (Future context)
    FEAT_LOOKAHEAD = [
        "u_in_next1",  # t+1
        "u_in_next2",  # t+2
        "u_in_next3",  # t+3
        "u_in_next4",  # t+4
        "u_in_diff_next1",  # Derivative at t+1
    ]

    # 3. Static Physics & Interactions
    FEAT_STATIC = [
        "R",
        "C",
        "R__u_in",  # Interaction: R * u_in
        "area__C",  # Interaction: area / C
    ]

    # Combined Feature List (Order matters for tensor construction)
    FEATURE_COLS = FEAT_DYNAMIC + FEAT_LOOKAHEAD + FEAT_STATIC

    # Input Dimension for the Model
    INPUT_DIM = len(FEATURE_COLS)

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print("=" * 40)
        print(f"CONFIG: {cls.__name__}")
        print("=" * 40)
        print(f"Device: {cls.DEVICE}")
        print(f"Debug Mode: {cls.DEBUG}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Epochs: {cls.EPOCHS}")
        print(f"Input Features ({cls.INPUT_DIM}): {cls.FEATURE_COLS}")
        print("=" * 40)
