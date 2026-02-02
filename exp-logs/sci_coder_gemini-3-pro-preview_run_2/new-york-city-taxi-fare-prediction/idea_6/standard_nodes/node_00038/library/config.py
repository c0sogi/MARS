import os

# -----------------------------------------------------------------------------
# Paths and Directories
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/optimized"
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure necessary directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Data Constraints & Sanitization
# -----------------------------------------------------------------------------
# Bounding box for NYC and surrounding areas (including airports)
# Used to clamp/filter coordinates to remove GPS errors/outliers
# Coordinates chosen to cover Manhattan, Queens, Brooklyn, Bronx, Staten Island, JFK, LGA, EWR
NYC_BOUNDING_BOX = {
    "lon_min": -74.50,
    "lon_max": -72.80,
    "lat_min": 40.50,
    "lat_max": 41.80,
}

# -----------------------------------------------------------------------------
# Feature Engineering Configuration
# -----------------------------------------------------------------------------
# Spatial Discretization
# Precision 6 is approx 1.2km x 0.6km, suitable for neighborhood-level grouping
GEOHASH_PRECISION = 6

# Target Encoding Parameters
# Used to calculate smoothed average fares for Route (Pickup_Hash -> Dropoff_Hash)
SMOOTHING_PARAMS = {
    "k_folds": 5,  # Number of folds for K-Fold Mean Encoding to prevent leakage
    "smoothing": 20,  # Smoothing factor (m) for Bayesian smoothing: (n*mean + m*global)/(n+m)
    "min_samples_leaf": 5,  # Minimum samples required to rely on the group mean
}

# -----------------------------------------------------------------------------
# Model Hyperparameters (XGBoost)
# -----------------------------------------------------------------------------
XGB_PARAMS = {
    # Objective Function
    # Switched to reg:squarederror to align with RMSE metric (Cite Lesson 00037).
    # Data sanitization handles the outliers that previously required robust loss.
    "objective": "reg:squarederror",
    # Evaluation Metric
    "eval_metric": "rmse",
    # Tree Architecture
    "n_estimators": 10000,  # High cap, controlled by early stopping
    "learning_rate": 0.01,  # Low LR for better convergence
    "max_depth": 8,  # Deeper trees to capture complex spatial boundaries
    "min_child_weight": 10,  # Regularization to prevent overfitting on rare routes
    "subsample": 0.7,  # Row sampling
    "colsample_bytree": 0.7,  # Feature sampling
    # Computation / Hardware
    "device": "cuda",  # Leverage NVIDIA A100 GPU
    "tree_method": "hist",  # Histogram-based algorithm for speed
    "n_jobs": 12,  # Number of CPU threads
    "random_state": 42,
}

# -----------------------------------------------------------------------------
# Training Loop Configuration
# -----------------------------------------------------------------------------
TRAIN_PARAMS = {
    "early_stopping_rounds": 100,  # Stop if validation RMSE doesn't improve
    "verbose_eval": 100,  # Print metrics every 100 rounds
    "random_state": 42,
    "test_size": 0.2,  # Fallback if metadata split isn't used
}
