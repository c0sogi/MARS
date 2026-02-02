import os
import torch


class Config:
    """
    Configuration class for the Hybrid Transformer-LSTM Ventilator Pressure Prediction.
    """

    # --- General ---
    SEED = 42
    DEBUG = False  # Set True to use a small subset of data for debugging
    EXP_NAME = "idea_6"
    OUTPUT_DIR = f"./working/{EXP_NAME}/"

    # --- Data Paths ---
    TRAIN_PATH = "./metadata/train.csv"
    VAL_PATH = "./metadata/val.csv"
    TEST_PATH = "./metadata/test.csv"
    SAMPLE_SUB_PATH = "./input/sample_submission.csv"

    # --- Data Processing ---
    SEQ_LEN = 80  # Fixed length of a breath sequence

    # Features to be used in the model
    # Continuous features (will be normalized)
    # Includes raw controls, time, and equation-driven engineered features
    CONT_FEATURES = [
        "time_step",
        "u_in",
        "u_out",
        "R",  # Treated as continuous
        "C",  # Treated as continuous
        "u_in_R",  # Interaction feature
        "u_in_C",  # Interaction feature
        "area",  # Cumulative volume (integral of u_in)
        "pressure_approx",  # Theoretical pressure based on Equation of Motion
        "u_in_cumsum",  # Cumulative sum of u_in
        "u_in_lag1",  # Lag features to capture immediate history
        "u_in_lag2",
        "u_in_diff1",  # Derivative features
        "u_in_diff2",
        "u_out_lag1",
        "u_out_lag2",
    ]

    # Categorical features (will be passed through Embedding layers)
    CAT_FEATURES = []

    # Target variable
    TARGET = "pressure"

    # --- Model Architecture (Multi-Scale CNN-LSTM) ---
    # CNN Stem
    CNN_FILTERS = 64  # Filters per branch

    # Residual Bi-LSTM
    LSTM_HIDDEN = 512
    LSTM_LAYERS = 4
    LSTM_BIDIRECTIONAL = True
    LSTM_DROPOUT = 0.1

    # Head
    FC_HIDDEN = 256

    # --- Training ---
    EPOCHS = 100
    BATCH_SIZE = 512  # Large batch size for A100
    LR = 1e-3  # Max learning rate for OneCycleLR
    WEIGHT_DECAY = 1e-4
    PATIENCE = 15  # Early stopping patience
    CLIP_GRAD = 1.0  # Gradient clipping value

    # Loss
    # We use Masked L1 Loss (MAE), ignoring the expiratory phase (u_out=1)

    # Scheduler (OneCycleLR)
    PCT_START = 0.1  # Percentage of training to increase LR
    DIV_FACTOR = 25  # Initial LR = Max LR / DIV_FACTOR
    FINAL_DIV_FACTOR = 1000  # Final LR = Initial LR / FINAL_DIV_FACTOR

    # --- Hardware ---
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """Ensures the output directory exists."""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)


# Initialize setup
Config.setup()
