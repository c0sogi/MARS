import os

# ==========================================
# Path Configuration
# ==========================================
INPUT_DIR = "./metadata"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
VAL_PATH = os.path.join(INPUT_DIR, "validation.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")

# Working directory for caching processed data
WORKING_DIR = "./working/idea_24"
os.makedirs(WORKING_DIR, exist_ok=True)
CACHE_DIR = WORKING_DIR

# Submission directory
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Hyperparameters
# ==========================================
# Training
SEED = 42
BATCH_SIZE = 128  # Stabilized Critical Mass Regime
EPOCHS = 80  # Budget
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
MAX_GRAD_NORM = 1.0  # Gradient Clipping

# Model Architecture: Kinematically-Augmented Residual-Hybrid (KARH-Net)
HIDDEN_DIM = 1024  # Wide-Latent Integration
LSTM_UNITS = 512  # High-Capacity Bidirectional LSTM
LSTM_LAYERS = 3  # Depth for numerical integration
CNN_CHANNELS = [64, 128, 256, 512]  # Deep Residual Dense TCN
KERNEL_SIZE = 9  # Geometry for smoothing
DROPOUT = 0.1

# ==========================================
# Feature Engineering Configuration
# ==========================================
# Columns to exclude from the model input
EXCLUDED_FEATURES = ["id", "breath_id", "pressure", "time_step"]

# Configuration for the feature engineering pipeline
FEATURE_GENERATION_CONFIG = {
    "leads": 4,  # Lookahead context: u_in at t+1...t+4
    "lags": 0,  # No raw lags, using explicit derivatives
    "diffs": 2,  # Backward derivatives (Velocity, Acceleration)
    "fwd_diffs": 1,  # Forward derivatives (Intent)
    "compute_physics": True,  # Area, dt
    "compute_interactions": True,  # R*u_in, Area/C
}

# Final list of features expected by the model
# This defines the input dimension and order
MODEL_FEATURES = [
    "u_in",  # Position
    "u_out",  # Valve State
    "R",  # Resistance
    "C",  # Compliance
    "u_in_diff1",  # Backward Velocity (Momentum)
    "u_in_diff2",  # Backward Acceleration
    "u_in_fwd1",  # Forward Velocity (Intent)
    "u_in_lead1",  # Lookahead t+1
    "u_in_lead2",  # Lookahead t+2
    "u_in_lead3",  # Lookahead t+3
    "u_in_lead4",  # Lookahead t+4
    "area",  # Volume (Integral)
    "time_step_diff",  # dt
    "R_u_in",  # Interaction: Resistive Pressure component
    "area_C",  # Interaction: Elastic Pressure component
]

# Input dimension for the model
INPUT_DIM = len(MODEL_FEATURES)
