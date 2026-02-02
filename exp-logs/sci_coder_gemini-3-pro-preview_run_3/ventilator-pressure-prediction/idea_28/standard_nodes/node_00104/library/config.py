import os


class Config:
    # --------------------------------------------------------------------------
    # Paths & Directories
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_28"
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # File Paths
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Hyperparameters
    # --------------------------------------------------------------------------
    SEED = 42
    BATCH_SIZE = 128
    EPOCHS = 80

    # Optimization
    LEARNING_RATE = 1e-3  # Initial LR for ReduceLROnPlateau
    WEIGHT_DECAY = 1e-4  # Low weight decay for regression stability
    CLIP_GRAD_NORM = 1.0

    # Scheduler (ReduceLROnPlateau)
    SCHEDULER_PATIENCE = 5
    SCHEDULER_FACTOR = 0.5
    MIN_LR = 1e-6

    # --------------------------------------------------------------------------
    # Model Architecture: PCDRH-Net
    # --------------------------------------------------------------------------
    # Branch 1: Deep Residual Dense TCN (The Resistive Stream)
    # Models high-frequency, derivative-dependent dynamics
    TCN_KERNEL_SIZE = 9
    TCN_DILATION = 1  # Dense convolution for high fidelity
    TCN_CHANNELS = 64
    TCN_LAYERS = 6  # Deep stack of Residual Dense Blocks
    TCN_DROPOUT = 0.1

    # Branch 2: High-Capacity Bidirectional LSTM (The Elastic Stream)
    # Models low-frequency, integral-dependent dynamics
    LSTM_HIDDEN_SIZE = 512
    LSTM_LAYERS = 3
    LSTM_BIDIRECTIONAL = True

    # Fusion Head: Wide-Latent Integration
    FUSION_HIDDEN_SIZE = 1024

    # --------------------------------------------------------------------------
    # Data Pipeline & Features
    # --------------------------------------------------------------------------
    # Stream A: Model Features (Scaled)
    # Processed with RobustScaler. Includes Kinematics, Physical State, and Explicit Physics.
    STREAM_A_FEATURES = [
        "R",
        "C",
        "u_in",
        "u_in_diff",  # Backward Velocity: u_in(t) - u_in(t-1)
        "u_in_lead1",  # Forward Lookahead t+1
        "u_in_lead2",  # Forward Lookahead t+2
        "u_in_lead3",  # Forward Lookahead t+3
        "u_in_lead4",  # Forward Lookahead t+4
        "area",  # Numerical Integration of Volume: sum(u_in * dt)
        "dt",  # Explicit Time-Delta
        "area_div_C",  # Interaction term: Area / C (Volume / Compliance)
        "u_out",  # Retained to explain expiratory drops to LSTM
    ]

    # Stream B: Logic/Mask Features (Raw)
    # No Scaling. Used for Logic-Gated Masked L1 Loss.
    STREAM_B_FEATURES = ["u_out"]

    # Target and ID columns
    TARGET_COL = "pressure"
    ID_COL = "id"
    BREATH_ID_COL = "breath_id"

    # Exclusions (as per "Implicit Physics" and "Pointwise Stem" lessons)
    # raw 'time_step' and Positional Encodings are excluded from Stream A
    EXCLUDED_COLS = ["time_step"]
