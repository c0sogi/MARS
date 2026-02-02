import os
import torch


class Config:
    """
    Configuration class for the RNA Degradation Prediction task.
    Implements the settings for the Hybrid GNN-BiLSTM architecture.
    """

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    METADATA_DIR = "./metadata"
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Working directory for intermediate files and caching
    WORKING_DIR = "./working/idea_2"
    CACHE_DIR = WORKING_DIR

    # Model checkpoint location
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "gnn_bilstm_model.pth")

    # Final submission output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Create necessary directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Dataset Constants
    # =========================================================================
    SEQ_LENGTH = 107
    SEQ_SCORED = 68

    # Feature Dimensions (One-Hot Encoding)
    # Sequence: {A, G, U, C} -> 4
    # Predicted Loop Type: {S, M, I, B, H, E, X} -> 7
    # Structure: {., (, )} -> 3
    NUM_NODE_FEATURES = 4 + 7 + 3  # Total: 14

    # Target Columns
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    NUM_TARGETS = 5

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    # Graph Neural Network (Spatial Encoder)
    GNN_HIDDEN_DIM = 128
    NUM_GNN_LAYERS = 3
    GNN_DROPOUT = 0.3

    # BiLSTM (Sequential Refinement)
    LSTM_HIDDEN_DIM = 128
    NUM_LSTM_LAYERS = 2
    LSTM_DROPOUT = 0.3

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10

    # Hardware settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
