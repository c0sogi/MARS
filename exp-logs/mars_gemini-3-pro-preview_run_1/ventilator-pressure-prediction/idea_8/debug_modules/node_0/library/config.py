import os
import torch


class Config:
    """
    Central configuration for the Ventilator Pressure Prediction pipeline.
    Implements the 'Dual-Gated Multi-Scale CNN-LSTM with Physics-Fidelity Features' strategy.
    """

    # --- General ---
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    EXP_ID = "idea_8"

    # --- Hardware ---
    # 12 vCPUs and 1 A100 GPU available
    NUM_WORKERS = 8
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Directories ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = os.path.join("./working", EXP_ID)
    SUBMISSION_DIR = "./submission"

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # --- File Paths ---
    # Metadata (Split by breath_id)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Cache Files (using .npy for efficiency and to avoid pickle issues)
    # These paths are used by the dataset module to store processed tensors
    CACHE_TRAIN_X = os.path.join(WORKING_DIR, "train_x.npy")
    CACHE_TRAIN_Y = os.path.join(WORKING_DIR, "train_y.npy")
    CACHE_VAL_X = os.path.join(WORKING_DIR, "val_x.npy")
    CACHE_VAL_Y = os.path.join(WORKING_DIR, "val_y.npy")
    CACHE_TEST_X = os.path.join(WORKING_DIR, "test_x.npy")
    CACHE_TEST_IDS = os.path.join(WORKING_DIR, "test_ids.npy")

    # Scaler Attributes (Center and Scale for RobustScaler)
    CACHE_SCALER_CENTER = os.path.join(WORKING_DIR, "scaler_center.npy")
    CACHE_SCALER_SCALE = os.path.join(WORKING_DIR, "scaler_scale.npy")

    # Model & Output
    MODEL_PATH = os.path.join(WORKING_DIR, "model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Data Parameters ---
    SEQ_LEN = 80  # Fixed length of a breath

    # Feature Engineering Flags
    # Strategy: Physics-Fidelity Features + Lags
    USE_PHYSICS_FEATURES = True
    USE_LAG_FEATURES = True
    LAG_STEPS = [1, 2, 3, 4]

    # --- Training Hyperparameters ---
    # Strategy: Extended optimization horizon (35 epochs) with large batch size
    EPOCHS = 35
    BATCH_SIZE = 512

    # Optimization (AdamW + OneCycleLR)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    MAX_GRAD_NORM = 1000.0

    # Scheduler
    PCT_START = 0.3
    DIV_FACTOR = 25.0
    FINAL_DIV_FACTOR = 1000.0

    # --- Model Architecture ---
    # Strategy: Dual-Gated Multi-Scale CNN-LSTM

    # Stem: Multi-Scale CNN
    CNN_KERNELS = [3, 5, 7]
    CNN_FILTERS = 64

    # Backbone: Dual-Gated Residual Bi-LSTM
    LSTM_HIDDEN_DIM = 256
    LSTM_LAYERS = 4
    BIDIRECTIONAL = True
    DROPOUT = 0.1

    # Gating & Head
    SE_RATIO = 16  # Reduction ratio for Squeeze-and-Excitation global gate
    HEAD_HIDDEN_DIM = 128

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print(f"=== Configuration: {cls.EXP_ID} ===")
        print(f"Device: {cls.DEVICE}")
        print(f"Batch Size: {cls.BATCH_SIZE}, Epochs: {cls.EPOCHS}")
        print(
            f"Model: {cls.LSTM_LAYERS}x LSTM (Dim {cls.LSTM_HIDDEN_DIM}) + CNN {cls.CNN_KERNELS}"
        )
        print(f"Physics Features: {cls.USE_PHYSICS_FEATURES}")
        print(f"Debug Mode: {cls.DEBUG}")
        print("====================================")
