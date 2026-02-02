import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from scipy.stats import pearsonr
import warnings

# Import provided library modules
from library import config
from library import preprocessing
from library import model as lib_model

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Setup
    set_seed(config.SEED)

    # 2. Data Loading
    # Uses the provided preprocessing pipeline which handles float64 conversion,
    # Yeo-Johnson transformation, and Standard Scaling.
    # load_cached_data=True ensures we use the artifacts if available.
    print("Loading and preprocessing data...")
    X_train, y_train, X_val, y_val, X_test, test_ids = (
        preprocessing.get_preprocessed_data(load_cached_data=True)
    )

    # 3. Model Training (Validation Split)
    print("Training model on training set...")
    clf = lib_model.DualPrecisionOAS()
    clf.fit(X_train, y_train)

    # 4. Validation Inference
    print("Running inference on validation set...")
    val_probs = clf.predict_proba(X_val)

    # 5. Validation Metric
    # Calculate Multi-class Log Loss
    # We explicitly provide labels to ensure correct mapping between probabilities and classes
    metric = log_loss(y_val, val_probs, labels=clf.classes_)

    # REQUIRED OUTPUT: Print the final validation metric
    print(f"Final Validation Metric: {metric}")

    # 6. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Map class labels to indices to extract the probability of the true class
    class_to_idx = {cls: i for i, cls in enumerate(clf.classes_)}
    y_val_indices = np.array([class_to_idx[y] for y in y_val])

    # Extract probability assigned to the true class
    # val_probs shape: (n_samples, n_classes)
    true_class_probs = val_probs[np.arange(len(y_val)), y_val_indices]

    # Calculate Error Magnitude (Log Loss contribution per sample)
    # Clip to avoid log(0)
    epsilon = 1e-15
    true_class_probs_clipped = np.maximum(true_class_probs, epsilon)
    error_magnitude = -np.log(true_class_probs_clipped)

    # Calculate correlation between each feature and the error magnitude
    correlations = []
    n_features = X_val.shape[1]

    for i in range(n_features):
        feature_values = X_val[:, i]
        # Skip constant features to avoid warnings/NaNs
        if np.std(feature_values) == 0:
            corr = 0.0
        else:
            corr, _ = pearsonr(feature_values, error_magnitude)

        feature_name = config.FEATURE_COLS[i]
        correlations.append((feature_name, corr))

    # Sort by absolute correlation strength (descending)
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 features correlated with error magnitude:")
    for name, corr in correlations[:5]:
        print(f"{name}: {corr:.4f}")

    # 7. Submission Generation
    # Threshold defined in task requirements
    THRESHOLD = 1.2136771218566717e-09

    if metric < THRESHOLD:
        print(
            f"\nValidation metric ({metric}) is below threshold ({THRESHOLD}). Generating submission..."
        )

        # Combine Train and Validation sets for full training
        # This maximizes the N for the OAS estimator
        X_full = np.concatenate([X_train, X_val], axis=0)
        y_full = np.concatenate([y_train, y_val], axis=0)

        print("Retraining on full dataset...")
        full_clf = lib_model.DualPrecisionOAS()
        full_clf.fit(X_full, y_full)

        print("Predicting on test set...")
        test_probs = full_clf.predict_proba(X_test)

        # Format Submission
        submission_df = pd.DataFrame(test_probs, columns=full_clf.classes_)
        # Insert ID column at the beginning
        submission_df.insert(0, config.ID_COL, test_ids.astype(int))

        submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(
            f"\nValidation metric ({metric}) is not below threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
