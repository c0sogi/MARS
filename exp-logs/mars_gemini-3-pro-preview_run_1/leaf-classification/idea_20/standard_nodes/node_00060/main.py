import sys
import os
import numpy as np
from sklearn.metrics import log_loss

# Ensure the current directory is in the path for module imports
sys.path.append(os.getcwd())

from library.config import SEED, FEATURE_COLS
from library.utils import set_seed, save_submission
from library.data_loader import load_datasets
from library.preprocessing import get_transformed_data
from library.model import StabilizedOASDiscriminant


def main():
    # 1. Initialization and Seeding
    set_seed(SEED)

    # 2. Data Loading
    # Uses caching to speed up loading if available
    X_train_raw, y_train, X_val_raw, y_val, X_test_raw, test_ids, classes = (
        load_datasets(load_cached_data=True)
    )

    # 3. Preprocessing
    # Applies Yeo-Johnson + StandardScaler. Fits on Train, transforms all.
    # Enforces float64 precision.
    X_train, X_val, X_test = get_transformed_data(
        X_train_raw, X_val_raw, X_test_raw, load_cached_data=True
    )

    # 4. Model Training
    # Initialize the Stabilized OAS Discriminant
    model = StabilizedOASDiscriminant()
    model.fit(X_train, y_train)

    # 5. Validation Inference
    val_probs = model.predict_proba(X_val)

    # Calculate Validation Metric (Multi-class Log Loss)
    # We explicitly provide labels to ensure correct mapping
    metric = log_loss(y_val, val_probs, labels=model.classes_)
    print(f"Final Validation Metric: {metric}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")

    # Calculate per-sample log loss
    # Map string labels to integer indices matching model.classes_
    class_to_idx = {cls: i for i, cls in enumerate(model.classes_)}
    y_val_indices = np.array([class_to_idx[y] for y in y_val])

    # Extract probability assigned to the true class
    # Clip to avoid log(0), matching the metric definition logic
    eps = 1e-15
    val_probs_clipped = np.clip(val_probs, eps, 1 - eps)
    true_class_probs = val_probs_clipped[np.arange(len(y_val)), y_val_indices]

    # Error magnitude = negative log likelihood
    sample_errors = -np.log(true_class_probs)

    # Calculate correlation between error magnitude and features
    correlations = []
    for i in range(X_val.shape[1]):
        feature_values = X_val[:, i]
        # Handle constant features to avoid NaN correlation
        if np.std(feature_values) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(sample_errors, feature_values)[0, 1]
        correlations.append((i, corr))

    # Sort by absolute correlation strength
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features Correlated with Error Magnitude:")
    for idx, corr in correlations[:5]:
        feat_name = FEATURE_COLS[idx]
        print(f"{feat_name}: {corr:.4f}")

    # 7. Submission Generation
    # Strict threshold check
    threshold = 1.2136771218566717e-09

    if metric < threshold:
        print(
            f"\nValidation metric ({metric}) meets threshold ({threshold}). Generating submission..."
        )
        test_probs = model.predict_proba(X_test)
        submission_path = "./submission/submission.csv"
        save_submission(test_ids, test_probs, model.classes_, submission_path)
    else:
        print(
            f"\nValidation metric ({metric}) does not meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
