import os
import torch


class Config:
    """
    Central configuration for the Ventilator Pressure Prediction task.
    Implements the settings for the Deeply Supervised Physics-Injected Hybrid CNN-LSTM-FFN.
    """

    # --------------------------------------------------------------------------
    # 1. Paths & Directories
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_12"
    SUBMISSION_DIR = "./submission"

    # Ensure necessary writeable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Data File Paths (using metadata splits)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Submission Output Path
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # 2. Data & Feature Engineering
    # --------------------------------------------------------------------------
    SEED = 42
    SEQ_LEN = 80  # Fixed breath length

    # Column Names
    ID = "id"
    BREATH_ID = "breath_id"
    TIME_STEP = "time_step"
    PRESSURE = "pressure"
    U_IN = "u_in"
    U_OUT = "u_out"
    R = "R"
    C = "C"

    # Engineered Feature Names
    DT = "dt"
    AREA = "area"  # u_in * dt
    VOLUME = "volume"  # cumsum(area)
    U_IN_LAG1 = "u_in_lag1"
    U_IN_LAG2 = "u_in_lag2"
    U_IN_LAG3 = "u_in_lag3"
    U_IN_LAG4 = "u_in_lag4"
    U_IN_DIFF1 = "u_in_diff1"
    U_IN_DIFF2 = "u_in_diff2"
    R_U_IN = "R_u_in"  # u_in * R (Interaction)
    VOL_DIV_C = "vol_div_C"  # volume / C (Interaction)

    # Full Input Feature List for the Model
    # These features must be present in the processed tensor
    INPUT_FEATURES = [
        TIME_STEP,
        U_IN,
        U_OUT,
        R,
        C,
        U_IN_LAG1,
        U_IN_LAG2,
        U_IN_LAG3,
        U_IN_LAG4,
        U_IN_DIFF1,
        U_IN_DIFF2,
        VOLUME,
        R_U_IN,
        VOL_DIV_C,
    ]

    # Features to be re-injected at every LSTM block (Physics Context)
    CONTEXT_FEATURES = [R, C, R_U_IN, VOL_DIV_C]

    NUM_FEATURES = len(INPUT_FEATURES)
    NUM_CONTEXT_FEATURES = len(CONTEXT_FEATURES)

    # --------------------------------------------------------------------------
    # 3. Model Architecture
    # --------------------------------------------------------------------------
    # Deeply Supervised Physics-Injected Hybrid CNN-LSTM-FFN
    HIDDEN_DIM = 512
    NUM_BLOCKS = 4
    STEM_KERNEL_SIZES = [3, 5, 7]  # Multi-scale Inception-like stem
    DROPOUT = 0.1
    USE_BIDIRECTIONAL = True

    # Deep Supervision
    AUX_BLOCK_INDEX = 1  # Attach aux head after the 2nd block (0-indexed)

    # --------------------------------------------------------------------------
    # 4. Training Hyperparameters
    # --------------------------------------------------------------------------
    EPOCHS = 30
    BATCH_SIZE = 512

    # Optimizer (AdamW) & Scheduler (OneCycleLR)
    LR_MAX = 1e-3
    WEIGHT_DECAY = 1e-2
    PCT_START = 0.3  # Percentage of training to increase LR

    # Loss
    AUX_LOSS_WEIGHT = 0.3  # Weight for the auxiliary head loss

    # --------------------------------------------------------------------------
    # 5. Hardware & Performance
    # --------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    PIN_MEMORY = True
