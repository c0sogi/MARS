import os
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder

# Import from provided libraries
from library.config import SEED, ALL_FEATURES, SUBMISSION_DIR, FLOAT_PRECISION
from library.data_loader import load_and_process_data
from library.preprocessing import get_preprocessed_data
from library.model import RatioProjectedOAS
from library.evaluation import compute_log_loss, generate_submission_file


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def perform_failure_analysis(model, X_val, y_val, feature_names):
    """
    Analyzes which features correlate with prediction errors.
    """
    print("\nPerforming Failure Analysis...")

    # 1. Calculate per-sample loss
    # Get probabilities
    probs = model.predict_proba(X_val)

    # Map string labels to indices
    le = LabelEncoder()
    le.classes_ = model.classes_
    y_indices = le.transform(y_val)

    # Extract probability assigned to the true class
    # Clip to avoid log(0)
    prob_true = probs[np.arange(len(y_val)), y_indices]
    prob_true = np.clip(prob_true, 1e-15, 1.0)

    # Calculate loss (negative log likelihood)
    sample_losses = -np.log(prob_true)

    # 2. Compute correlation with features
    # X_val is a numpy array, feature_names corresponds to columns
    n_features = X_val.shape[1]
    correlations = []

    for i in range(n_features):
        # Calculate correlation between feature values and loss
        # Handle potential constant features (std=0) to avoid NaN
        feat_values = X_val[:, i]
        if np.std(feat_values) > 1e-9 and np.std(sample_losses) > 1e-9:
            corr = np.corrcoef(feat_values, sample_losses)[0, 1]
        else:
            corr = 0.0
        correlations.append((feature_names[i], corr))

    # 3. Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 10 features correlated with error magnitude:")
    for name, corr in correlations[:10]:
        print(f"  {name}: {corr:.4f}")


def main():
    # 1. Setup
    set_seed(SEED)

    # 2. Load Data
    # load_cached_data=True allows using the artifacts from previous runs if available
    print("Loading data...")
    (
        (X_train_raw, y_train, train_ids),
        (X_val_raw, y_val, val_ids),
        (X_test_raw, test_ids),
    ) = load_and_process_data(load_cached_data=True)

    # 3. Preprocess
    # Applies Yeo-Johnson and StandardScaling
    print("Preprocessing data...")
    X_train, X_val, X_test = get_preprocessed_data(
        X_train_raw, X_val_raw, X_test_raw, load_cached_data=True
    )

    # 4. Train Model
    print("Training Ratio-Projected OAS Discriminant...")
    model = RatioProjectedOAS()
    model.fit(X_train, y_train)

    # 5. Validation
    print("Validating...")
    val_probs = model.predict_proba(X_val)

    # Compute Metric
    # Note: compute_log_loss handles the specific clipping and rescaling requirements
    val_loss = compute_log_loss(y_val, val_probs, model.classes_)

    # REQUIRED: Print the validation metric in the exact format
    print(f"Final Validation Metric: {val_loss}")

    # 6. Failure Analysis
    perform_failure_analysis(model, X_val, y_val, ALL_FEATURES)

    # 7. Submission
    # Strict threshold check as per requirements
    THRESHOLD = 3.3382359570696616e-14

    if val_loss < THRESHOLD:
        print(
            f"\nValidation metric ({val_loss}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Predict on Test Set
        test_probs = model.predict_proba(X_test)

        # Generate Submission File
        submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        generate_submission_file(test_ids, test_probs, model.classes_, submission_path)
    else:
        print(
            f"\nValidation metric ({val_loss}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
