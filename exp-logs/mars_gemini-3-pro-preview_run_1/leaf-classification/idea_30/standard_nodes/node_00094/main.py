import numpy as np
import pandas as pd
import warnings
import sys
from sklearn.preprocessing import LabelEncoder

# Import custom library modules
from library.config import SEED
from library.data_loader import load_datasets
from library.preprocessing import get_preprocessed_data
from library.model import RobustOASClassifier
from library.utils import compute_log_loss, save_submission

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)


def perform_failure_analysis(X_val, y_val, val_probs, classes):
    """
    Analyzes prediction errors on the validation set.
    Calculates the correlation between error magnitude and input features.
    """
    print("\nPerforming Failure Analysis...")

    # Encode y_val to indices matching the probability matrix columns
    le = LabelEncoder()
    le.fit(classes)
    y_val_indices = le.transform(y_val)

    # Calculate error magnitude (Log Loss contribution per sample)
    # We clip probabilities to avoid log(0)
    prob_true = val_probs[np.arange(len(y_val)), y_val_indices]
    prob_true_clipped = np.clip(prob_true, 1e-15, 1.0 - 1e-15)
    error_magnitude = -np.log(prob_true_clipped)

    # Calculate correlation between each feature and the error magnitude
    n_features = X_val.shape[1]
    correlations = []

    for i in range(n_features):
        feature_vals = X_val[:, i]
        # Handle potential constant features which produce NaN correlation
        if np.std(feature_vals) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(feature_vals, error_magnitude)[0, 1]
        correlations.append(corr)

    correlations = np.array(correlations)

    # Identify top features associated with error
    # We look at absolute correlation to find strong relationships (positive or negative)
    top_indices = np.argsort(np.abs(correlations))[::-1][:5]

    print("Top 5 features correlated with error magnitude:")
    for idx in top_indices:
        print(f"Feature {idx} (corr: {correlations[idx]:.4f})")


def main():
    # 1. Setup
    set_seed(SEED)

    # 2. Load Data
    # We use load_cached_data=True to utilize any existing artifacts in ./working
    print("Loading datasets...")
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = load_datasets(
        load_cached_data=True
    )

    # 3. Preprocessing
    # Applies Yeo-Johnson and Standard Scaling in float64 precision
    print("Preprocessing data...")
    X_train_trans, X_val_trans, X_test_trans = get_preprocessed_data(
        X_train, X_val, X_test, load_cached_data=True
    )

    # 4. Model Training
    # Initialize the Robust-OAS Classifier
    print("Training RobustOASClassifier...")
    model = RobustOASClassifier()

    # Fit on the training set
    # The model internally handles class means, priors, and OAS covariance estimation
    model.fit(X_train_trans, y_train)

    # 5. Validation
    print("Validating model...")
    # Predict probabilities (float64)
    val_probs = model.predict_proba(X_val_trans)

    # Compute Metric
    val_loss = compute_log_loss(y_val, val_probs, model.classes_)

    # PRINT FINAL METRIC (Required Format)
    print(f"Final Validation Metric: {val_loss}")

    # 6. Failure Analysis
    perform_failure_analysis(X_val_trans, y_val, val_probs, model.classes_)

    # 7. Submission
    # Threshold defined in the task
    THRESHOLD = 1.2136771218566717e-09

    if val_loss < THRESHOLD:
        print(
            f"\nValidation metric ({val_loss}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        test_probs = model.predict_proba(X_test_trans)
        save_submission(test_probs, test_ids, model.classes_)
    else:
        print(
            f"\nValidation metric ({val_loss}) does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
