import os
import torch

# Ensure the working directory exists immediately upon import
WORKING_DIR = "./working/idea_22"
os.makedirs(WORKING_DIR, exist_ok=True)


class Config:
    # --- Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = WORKING_DIR

    # --- Reproducibility ---
    SEED = 42

    # --- Data Processing ---
    # Window size for temporal context: t-5 to t+5 (11 frames total)
    WINDOW_SIZE = 5

    # --- Feature Definitions ---

    # Base Continuous Kinematic Features (per player)
    # These will be expanded over the temporal window
    KINEMATIC_FEATURES = [
        "x_position",
        "y_position",
        "speed",
        "acceleration",
        "sa",  # Signed acceleration
        "distance_traveled",  # 'distance' column in raw tracking (renamed to avoid confusion)
        "orientation_sin",
        "orientation_cos",
        "vx",
        "vy",
    ]

    # Computed Interaction Features (Relative between pair)
    # These will also be expanded over the temporal window
    INTERACTION_FEATURES = [
        "distance",  # Euclidean distance
        "log_distance",  # np.log1p(distance)
        "relative_speed",  # Magnitude of velocity difference
        "closing_speed",  # Project velocity onto distance vector
        "diff_x",
        "diff_y",
    ]

    # Categorical Features for Entity Embeddings
    # Used to provide role-based context
    CATEGORICAL_COLS = ["position", "team"]

    # Visual Features (Helmet Box Metrics)
    # Flattened wide feature vector derived from Max-Pooling strategy
    VISUAL_FEATURES = ["left", "width", "top", "height", "area"]

    # --- Model Architecture ---

    # Kinematic Stream (Gated Residual Backbone)
    KINEMATIC_INPUT_DIM = (len(KINEMATIC_FEATURES) * 2 + len(INTERACTION_FEATURES)) * (
        2 * WINDOW_SIZE + 1
    )
    KINEMATIC_HIDDEN_DIM = 512
    KINEMATIC_LAYERS = 4
    DROPOUT = 0.1

    # Visual Stream (Shallow Correction)
    VISUAL_INPUT_DIM = (
        len(VISUAL_FEATURES) * 2 * (2 * WINDOW_SIZE + 1)
    )  # *2 for two players
    VISUAL_HIDDEN_DIM = 64

    # Embeddings
    EMBEDDING_DIM = 8

    # --- Training ---
    BATCH_SIZE = 1024  # Large batch size for stability with wide features
    LEARNING_RATE = 1e-3
    EPOCHS = 20  # Max epochs, controlled by early stopping
    EARLY_STOPPING_PATIENCE = 3

    # --- Loss Function (Focal Loss) ---
    # Critical parameters for class imbalance
    ALPHA = 0.25
    GAMMA = 2.0

    # --- Hardware ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
