import numpy as np
import pandas as pd
import sys
import os

# Import necessary components from the provided library
from library.config import SEED, SUBMISSION_FILE_PATH
from library.preprocessing import get_preprocessed_data
from library.modeling import GlobalLDAModel
from library.utils import calculate_log_loss, save_submission


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)


def main():
    # 1. Setup
    set_seed(SEED)

    # 2. Data Loading
    # Load preprocessed data (PowerTransformed + Scaled)
    # This uses the caching mechanism defined in library.preprocessing
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, test_ids = get_preprocessed_data(
        load_cached_data=True
    )

    # 3. Model Training
    print("Initializing and training GlobalLDAModel...")
    model = GlobalLDAModel()
    model.fit(X_train, y_train)

    # 4. Validation Inference
    print("Performing validation inference...")
    val_probs = model.predict_proba(X_val)

    # 5. Metric Calculation
    # Calculate log loss using the provided utility which handles clipping and rescaling
    metric = calculate_log_loss(y_val, val_probs, model.classes_)

    # Print the metric in the required format with full precision
    print(f"Final Validation Metric: {metric}")

    # 6. Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Map string labels to indices
    class_map = {label: i for i, label in enumerate(model.classes_)}
    y_val_indices = np.array([class_map[label] for label in y_val])

    # Calculate per-sample error (negative log likelihood of the true class)
    # Clip probabilities to avoid log(0), consistent with the metric calculation
    eps = 1e-15
    # Row-wise normalization is already done implicitly by the model/metric logic,
    # but we ensure numerical stability here for analysis.
    val_probs_clipped = np.clip(val_probs, eps, 1 - eps)

    # Extract probability assigned to the true class
    true_class_probs = val_probs_clipped[np.arange(len(y_val)), y_val_indices]

    # Error magnitude
    errors = -np.log(true_class_probs)

    # Calculate correlation between features and error
    # X_val shape: (n_samples, n_features)
    # errors shape: (n_samples,)
    correlations = []
    n_features = X_val.shape[1]

    for i in range(n_features):
        feature_col = X_val[:, i]
        # Calculate Pearson correlation
        if np.std(feature_col) > 0 and np.std(errors) > 0:
            corr = np.corrcoef(feature_col, errors)[0, 1]
            correlations.append((i, corr))
        else:
            correlations.append((i, 0.0))

    # Sort by absolute correlation strength
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 features correlated with prediction error:")
    for idx, corr in correlations[:5]:
        print(f"Feature index {idx}: Correlation {corr:.4f}")

    # 7. Submission Generation
    # Strict threshold check
    threshold = 1.470544781593644e-08

    if metric < threshold:
        print(f"\nValidation metric {metric} meets threshold {threshold}.")
        print("Generating submission for test set...")

        test_probs = model.predict_proba(X_test)

        save_submission(
            ids=test_ids,
            class_labels=model.classes_,
            probabilities=test_probs,
            output_path=SUBMISSION_FILE_PATH,
        )
    else:
        print(f"\nValidation metric {metric} does not meet threshold {threshold}.")
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
