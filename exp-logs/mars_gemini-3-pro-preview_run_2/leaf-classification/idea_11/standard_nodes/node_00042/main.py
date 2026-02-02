import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

# Import provided library modules
from library.models import get_linear_branch, get_generative_branch
from library.ensemble import soft_vote, run_ensemble

# ==============================================================================
# CONFIGURATION
# ==============================================================================
RANDOM_SEED = 42
METADATA_DIR = "./metadata"
SUBMISSION_DIR = "./submission"
THRESHOLD_METRIC = 0.009092766987305665

# Set seeds for reproducibility
np.random.seed(RANDOM_SEED)


def load_split_data():
    """
    Loads training and validation data separately from metadata files
    to perform strict hold-out validation.
    """
    train_path = os.path.join(METADATA_DIR, "train.csv")
    val_path = os.path.join(METADATA_DIR, "val.csv")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError("Metadata files (train.csv, val.csv) not found.")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    # Identify feature columns (exclude metadata)
    exclude_cols = {"id", "species", "image_path"}
    feature_cols = [c for c in df_train.columns if c not in exclude_cols]
    feature_cols.sort()  # Ensure consistent order

    # Extract Features and Targets
    X_train_raw = df_train[feature_cols].values
    y_train_raw = df_train["species"].values

    X_val_raw = df_val[feature_cols].values
    y_val_raw = df_val["species"].values

    # Preprocessing
    # 1. Label Encoding
    # Fit on all unique species to ensure coverage (though stratification should handle this)
    all_species = np.unique(np.concatenate([y_train_raw, y_val_raw]))
    le = LabelEncoder()
    le.fit(all_species)

    y_train = le.transform(y_train_raw)
    y_val = le.transform(y_val_raw)

    # 2. Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    return X_train, y_train, X_val, y_val, le, feature_cols


def clip_probabilities(probs):
    """
    Clips probabilities to [1e-15, 1-1e-15] to avoid log(0) extremes.
    """
    epsilon = 1e-15
    probs = np.clip(probs, epsilon, 1 - epsilon)
    # Renormalize to ensure sum to 1 after clipping (optional but good practice)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


def perform_failure_analysis(X_val, y_val, probs, feature_names):
    """
    Analyzes correlation between error magnitude and input features.
    """
    print("\n--- Failure Analysis ---")

    # Calculate error magnitude per sample
    # Error = -log(p_true) (Log Loss contribution)
    # Or simply (1 - p_true)

    # Get probability assigned to the true class
    rows = np.arange(len(y_val))
    true_class_probs = probs[rows, y_val]

    # Error magnitude (Log Loss contribution)
    errors = -np.log(true_class_probs)

    correlations = []
    for i, feature_name in enumerate(feature_names):
        feature_values = X_val[:, i]
        # Calculate Pearson correlation
        corr, _ = pearsonr(feature_values, errors)
        if np.isnan(corr):
            corr = 0
        correlations.append((feature_name, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error Magnitude:")
    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.4f}")


def main():
    print("Starting Pipeline Execution...")

    # 1. Load Data for Validation
    print("Loading split data for validation...")
    X_train, y_train, X_val, y_val, le, feature_cols = load_split_data()
    print(f"Train Shape: {X_train.shape}, Val Shape: {X_val.shape}")

    # 2. Initialize Models
    print("Initializing models...")
    # Using smaller max_iter or n_jobs where applicable for speed if necessary,
    # but defaults in library are robust.
    linear_model = get_linear_branch(random_state=RANDOM_SEED)
    generative_model = get_generative_branch()

    # 3. Train on Split Train Set
    print("Training Linear Branch...")
    linear_model.fit(X_train, y_train)

    print("Training Generative Branch...")
    generative_model.fit(X_train, y_train)

    # 4. Validation Inference
    print("Running Validation Inference...")
    probs_linear = linear_model.predict_proba(X_val)
    probs_gen = generative_model.predict_proba(X_val)

    # 5. Ensemble Aggregation
    print("Aggregating predictions...")
    avg_probs = soft_vote([probs_linear, probs_gen])

    # Clip probabilities for metric calculation
    avg_probs_clipped = clip_probabilities(avg_probs)

    # 6. Calculate Metric
    val_log_loss = log_loss(y_val, avg_probs_clipped)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_log_loss}")

    # 7. Failure Analysis
    perform_failure_analysis(X_val, y_val, avg_probs_clipped, feature_cols)

    # 8. Conditional Submission
    # The prompt requires: "If and only if the final validation metric is lower than 0.009092766987305665"
    # Note: Lower log loss is better.
    # However, 0.009 is extremely low. If the threshold implies we must be BETTER (lower) than it:
    if val_log_loss < THRESHOLD_METRIC:
        print(
            f"\nValidation metric ({val_log_loss}) meets threshold ({THRESHOLD_METRIC})."
        )
        print("Proceeding to generate submission on full dataset...")

        # Use the library function to retrain on full data and generate submission
        # We set load_cached_data=False to ensure we process the full dataset correctly
        # if the cache was previously just the split or non-existent.
        run_ensemble(load_cached_data=False, random_state=RANDOM_SEED)

    else:
        print(
            f"\nValidation metric ({val_log_loss}) does not meet threshold ({THRESHOLD_METRIC})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
