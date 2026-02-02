import os
import sys
import numpy as np
import pandas as pd
import scipy.stats
import torch

# Import from provided libraries
from library.config import Config
from library.data_loader import LeafDataLoader
from library.preprocessing import FeatureScaler
from library.model import LeafEnsemble, generate_submission
from library.evaluation import compute_log_loss, evaluate_model


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def perform_failure_analysis(model, X_val, y_val, feature_names=None):
    """
    Analyzes model failures by correlating error magnitude with input features.
    """
    print("\nPerforming Failure Analysis...")

    # 1. Get Predictions
    y_pred = model.predict(X_val)  # Returns probabilities

    # 2. Calculate Per-Sample Log Loss (Error Magnitude)
    # Extract the probability assigned to the true class
    # y_val are indices [0, n_classes-1]
    # We gather the prob corresponding to the true index for each row
    row_indices = np.arange(len(y_val))
    true_class_probs = y_pred[row_indices, y_val]

    # Clip to avoid log(0)
    eps = 1e-15
    true_class_probs = np.clip(true_class_probs, eps, 1 - eps)

    # Error magnitude is negative log likelihood
    error_magnitude = -np.log(true_class_probs)

    # 3. Correlate Error with Features
    n_features = X_val.shape[1]
    correlations = []

    for i in range(n_features):
        feature_vals = X_val[:, i]
        # Handle constant features (std=0) to avoid nan correlation
        if np.std(feature_vals) < 1e-9:
            corr = 0.0
        else:
            corr, _ = scipy.stats.pearsonr(feature_vals, error_magnitude)
        correlations.append(corr)

    correlations = np.array(correlations)

    # 4. Report Top Correlations
    # Sort by absolute correlation
    top_indices = np.argsort(np.abs(correlations))[::-1][:5]

    print("Top 5 Features correlated with Error Magnitude:")
    for idx in top_indices:
        feat_name = f"Feature_{idx}"  # We don't have explicit col names in X matrix, using index
        print(f"  - {feat_name}: Correlation = {correlations[idx]:.4f}")


def main():
    # 1. Configuration and Setup
    set_seed(Config.RANDOM_SEED)

    # Detect GPU (Requirement check, though sklearn runs on CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Inference device detected: {device}")

    # 2. Load Data
    print("Loading data...")
    loader = LeafDataLoader()
    # Using cached data if available for speed
    data = loader.load_data(load_cached_data=True)

    X_train, y_train, train_ids = data["train"]
    X_val, y_val, val_ids = data["val"]
    X_test, test_ids = data["test"]
    encoder = data["encoder"]

    # 3. Preprocessing (Scaling)
    print("Scaling features...")
    scaler = FeatureScaler()
    X_train_scaled, X_val_scaled, X_test_scaled = scaler.scale_features(
        X_train, X_val, X_test, load_cached_data=True
    )

    # 4. Model Training
    print("Initializing and training model...")
    # Using LeafEnsemble (LR + LDA)
    model = LeafEnsemble()
    model.train(X_train_scaled, y_train, X_val_scaled, y_val)

    # 5. Validation
    print("Validating model...")
    val_loss = evaluate_model(model, X_val_scaled, y_val)

    # REQUIRED: Print Final Validation Metric with full precision
    print(f"Final Validation Metric: {val_loss}")

    # 6. Failure Analysis
    perform_failure_analysis(model, X_val_scaled, y_val)

    # 7. Generate Submission
    # Requirement: Only generate submission if val_loss is lower than 0.010187299388940634
    baseline_metric = 0.010187299388940634

    if val_loss < baseline_metric:
        print(
            f"Validation metric {val_loss} is better than baseline {baseline_metric}. Generating submission..."
        )

        # Retrain on Full Data (Train + Val) to maximize performance (Cite solution_lesson_node_00010)
        print("Retraining on combined Train + Validation set...")
        X_full = np.vstack((X_train_scaled, X_val_scaled))
        y_full = np.concatenate((y_train, y_val))

        # Re-initialize model to reset state
        final_model = LeafEnsemble()
        final_model.train(X_full, y_full)

        submission_path = Config.SUBMISSION_PATH

        # Use the library function to generate submission
        generate_submission(
            final_model, X_test_scaled, test_ids, encoder, submission_path
        )
    else:
        print(
            f"Validation metric {val_loss} did not improve upon baseline {baseline_metric}. Skipping submission."
        )

    print("Run complete.")


if __name__ == "__main__":
    main()
