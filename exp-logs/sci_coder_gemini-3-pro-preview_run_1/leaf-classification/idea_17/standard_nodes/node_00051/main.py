import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import provided library modules
from library.config import SUBMISSION_OUTPUT_FILE, FEATURE_COLS
from library.utils import seed_everything, log_loss_metric, format_submission
from library.preprocessor import get_preprocessed_data
from library.oas_discriminant import OASDiscriminant


def run():
    # 1. Setup and Reproducibility
    seed_everything()

    # 2. Data Loading and Preprocessing
    # Loads cached float32 data if available, otherwise runs the PrecisionPipeline
    print("Loading and preprocessing data...")
    X_train, y_train, X_val, y_val, X_test, test_ids, class_names = (
        get_preprocessed_data(load_cached_data=True)
    )

    # 3. Model Training
    # Initialize the Precision-Quantized Supervised OAS Discriminant
    print("Initializing Supervised OAS Discriminant...")
    model = OASDiscriminant()

    # Fit the model:
    # - Estimates means on Train
    # - Estimates covariance on Train residuals using OAS
    # - Quantizes parameters to float32
    print("Training model...")
    model.fit(X_train, y_train)

    # 4. Validation
    print("Performing validation inference...")
    # Predict probabilities on validation set (inference in float32)
    val_probs = model.predict_proba(X_val)

    # Compute Final Validation Metric
    # Uses the specific rescaling and clipping logic defined in utils
    val_metric = log_loss_metric(y_val, val_probs)
    print(f"Final Validation Metric: {val_metric}")

    # 5. Failure Analysis
    print("Performing failure analysis...")
    # Calculate error magnitude: 1.0 - probability assigned to the true class
    # y_val contains the true class indices
    rows = np.arange(len(y_val))
    true_class_probs = val_probs[rows, y_val]
    error_magnitudes = 1.0 - true_class_probs

    # Calculate correlation between error magnitude and each feature
    correlations = []
    n_features = X_val.shape[1]

    for i in range(n_features):
        feature_vals = X_val[:, i]
        # Check for constant features to avoid division by zero in correlation
        if np.std(feature_vals) > 1e-12:
            corr, _ = pearsonr(feature_vals, error_magnitudes)
            correlations.append((i, corr))
        else:
            correlations.append((i, 0.0))

    # Sort by absolute correlation (descending)
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 features correlated with error magnitude:")
    for i, corr in correlations[:5]:
        # Retrieve feature name safely
        feat_name = FEATURE_COLS[i] if i < len(FEATURE_COLS) else f"Feature_{i}"
        print(f"  {feat_name}: {corr:.6f}")

    # 6. Submission Generation
    # Strictly adhere to the threshold requirement
    threshold = 1.2136771218566717e-09

    if val_metric < threshold:
        print(
            f"Validation metric ({val_metric}) meets threshold ({threshold}). Generating submission..."
        )

        # Generate predictions for the test set
        test_probs = model.predict_proba(X_test)

        # Format and save the submission file
        format_submission(test_ids, test_probs, class_names, SUBMISSION_OUTPUT_FILE)
    else:
        print(
            f"Validation metric ({val_metric}) did not meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    run()
