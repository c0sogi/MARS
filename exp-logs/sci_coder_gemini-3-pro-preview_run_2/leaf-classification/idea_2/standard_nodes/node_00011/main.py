import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.data_processing import load_and_preprocess_data
from library.model_factory import create_hybrid_ensemble
from library.utils import calculate_metric, save_submission


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def perform_failure_analysis(X, y, y_pred_proba, classes):
    """
    Analyzes model failures by correlating feature values with prediction error.
    """
    print("\nPerforming Failure Analysis...")

    # 1. Calculate per-sample Log Loss
    # Get the predicted probability for the true class
    # y is the integer index of the class
    # We clip probabilities to avoid log(0)
    eps = Config.CLIP_EPSILON
    y_pred_clipped = np.clip(y_pred_proba, eps, 1 - eps)

    # Select the probability of the true class for each sample
    true_class_probs = y_pred_clipped[np.arange(len(y)), y]

    # Calculate Log Loss: -log(p_true)
    sample_losses = -np.log(true_class_probs)

    print(f"Mean Sample Loss: {np.mean(sample_losses):.6f}")
    print(f"Max Sample Loss: {np.max(sample_losses):.6f}")

    # 2. Correlate Error with Features
    n_features = X.shape[1]
    correlations = []

    for i in range(n_features):
        # Calculate Pearson correlation between feature values and loss
        # Handle cases where feature might be constant (std=0)
        if np.std(X[:, i]) > 0:
            corr, _ = pearsonr(X[:, i], sample_losses)
            correlations.append((i, corr))
        else:
            correlations.append((i, 0.0))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\nTop 5 Features correlated with Error Magnitude:")
    for idx, corr in correlations[:5]:
        print(f"  Feature Index {idx}: Correlation = {corr:.4f}")


def main():
    # 1. Setup
    set_seed(Config.RANDOM_SEED)
    print("Starting Runfile execution...")

    # 2. Load Data
    # Using cached data for speed as per instructions
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = (
        load_and_preprocess_data(load_cached_data=True)
    )

    print(
        f"Data Loaded: Train shape {X_train.shape}, Val shape {X_val.shape}, Test shape {X_test.shape}"
    )

    # 3. Initialize Model
    model = create_hybrid_ensemble()

    # 4. Train on Training Set
    print("Training model on training set...")
    model.fit(X_train, y_train)

    # 5. Validation Inference
    print("Evaluating on validation set...")
    # Sklearn models don't have explicit eval mode or GPU .to() methods in the same way PyTorch does,
    # but we ensure efficient execution via n_jobs=-1 in the model config.
    y_val_pred_proba = model.predict_proba(X_val)

    # Convert integer labels to string labels for metric calculation
    y_val_str = classes[y_val]

    # Calculate Metric
    val_score = calculate_metric(y_val_str, y_val_pred_proba, classes=classes)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {val_score}")

    # 6. Failure Analysis
    perform_failure_analysis(X_val, y_val, y_val_pred_proba, classes)

    # 7. Submission Logic
    threshold = 0.010187299388940634

    if val_score < threshold:
        print(
            f"\nValidation score ({val_score}) meets threshold ({threshold}). Proceeding to submission."
        )

        # Retrain on Combined Data (Train + Val) for maximum performance
        print("Retraining ensemble on combined Train + Validation data...")
        X_full = np.vstack((X_train, X_val))
        y_full = np.concatenate((y_train, y_val))

        model.fit(X_full, y_full)

        # Predict on Test Set
        print("Generating predictions for test set...")
        y_test_pred_proba = model.predict_proba(X_test)

        # Save Submission
        print(f"Saving submission to {Config.SUBMISSION_PATH}...")
        save_submission(test_ids, classes, y_test_pred_proba)
        print("Submission saved successfully.")

    else:
        print(
            f"\nValidation score ({val_score}) did not meet threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
