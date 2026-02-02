import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Add current directory to sys.path to ensure library imports work
sys.path.append(os.getcwd())

from library import config
from library import utils
from library import train_eval
from library.data_processing import DataHandler


def main():
    # 1. Setup
    # Set random seed for reproducibility
    utils.set_seed()

    # 2. Data Loading
    # Initialize DataHandler to manage data loading and preprocessing
    # load_cached_data=True allows using pre-computed .npy files if they exist
    print("Initializing data handler and loading data...")
    data_handler = DataHandler()
    X_train, y_train, X_val, y_val, X_test, ids_test = data_handler.get_processed_data(
        load_cached_data=True
    )

    # 3. Training
    # We use the full dataset for LightGBM to capture non-linear interactions effectively.
    # Cite solution_lesson_node_00001
    print(f"Training LightGBM model with full dataset...")
    model = train_eval.train_model(X_train, y_train, X_val, y_val, max_samples=None)

    # 4. Validation
    print("Performing validation on the full validation set...")
    # Predict probabilities for the positive class (state 1)
    # Note: The provided model is Scikit-Learn based (CPU), so we do not perform explicit GPU transfers.
    y_val_pred = model.predict_proba(X_val)[:, 1]

    # Calculate ROC AUC
    val_auc = roc_auc_score(y_val, y_val_pred)

    # Print the validation metric in the required format
    print(f"Final Validation Metric: {val_auc}")

    # 5. Failure Analysis
    print("Performing failure analysis...")
    # Calculate the absolute error for each validation sample
    errors = np.abs(y_val - y_val_pred)

    # Calculate correlation between each feature and the error magnitude
    n_features = X_val.shape[1]
    correlations = []

    print(f"Computing error correlations for {n_features} features...")
    for i in range(n_features):
        feature_col = X_val[:, i]

        # Check for constant features to avoid division by zero in correlation
        if np.std(feature_col) > 1e-9:
            # Compute Pearson correlation
            corr = np.corrcoef(feature_col, errors)[0, 1]
            if np.isnan(corr):
                corr = 0.0
        else:
            corr = 0.0

        correlations.append((i, corr))

    # Sort correlations by magnitude (descending)
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 10 features correlated with prediction error:")
    for idx, corr in correlations[:10]:
        print(f"Feature Index {idx}: Correlation = {corr:.6f}")

    # 6. Submission
    # Only submit if we meet the high performance threshold
    threshold = 0.9697562490846128
    if val_auc > threshold:
        print(
            f"Validation AUC {val_auc} exceeds threshold {threshold}. Generating submission..."
        )
        train_eval.predict_test(model, X_test, ids_test)
    else:
        print(
            f"Validation AUC {val_auc} does not meet threshold {threshold}. Skipping submission."
        )

    print("Runfile execution complete.")


if __name__ == "__main__":
    main()
