import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import from provided library
from library.config import SEED, SUBMISSION_PATH
from library.preprocessing import get_preprocessed_data
from library.model import OASDiscriminant
from library.utils import compute_log_loss, save_submission


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Setup
    set_seed(SEED)

    # 2. Load Data
    # The get_preprocessed_data function handles loading raw data,
    # applying the Iterative Gaussianization pipeline, and caching.
    # It returns ((X_train, y_train, ids_train), (X_val, y_val, ids_val), (X_test, ids_test))
    (train_data, val_data, test_data) = get_preprocessed_data(load_cached_data=True)

    X_train, y_train, ids_train = train_data
    X_val, y_val, ids_val = val_data
    X_test, ids_test = test_data

    # 3. Model Training
    # Initialize the OAS Discriminant Analysis model
    model = OASDiscriminant()

    # Fit on the training set
    model.fit(X_train, y_train)

    # 4. Validation
    # Predict probabilities on the validation set
    y_val_pred = model.predict_proba(X_val)

    # Compute the metric
    val_loss = compute_log_loss(y_val, y_val_pred, model.classes_)

    # REQUIRED OUTPUT: Print the final validation metric
    print(f"Final Validation Metric: {val_loss}")

    # 5. Failure Analysis
    print("\nStarting Failure Analysis...")

    # Calculate per-sample error (negative log likelihood of the true class)
    # First, map class labels to indices
    class_to_idx = {c: i for i, c in enumerate(model.classes_)}
    y_val_indices = np.array([class_to_idx[c] for c in y_val])

    # Extract probabilities of the true classes
    # y_val_pred is (n_samples, n_classes)
    # We use advanced indexing to get the prob of the true class for each sample
    p_correct = y_val_pred[np.arange(len(y_val)), y_val_indices]

    # Clip probabilities to avoid log(0) and match metric calculation stability
    epsilon = 1e-15
    p_correct_clipped = np.clip(p_correct, epsilon, 1 - epsilon)

    # Calculate loss per sample
    sample_losses = -np.log(p_correct_clipped)

    # Compute correlation between feature values and error magnitude
    correlations = []
    n_features = X_val.shape[1]

    for i in range(n_features):
        feature_values = X_val[:, i]
        # Check for constant features to avoid warnings/NaNs
        if np.std(feature_values) < 1e-12:
            corr = 0.0
        else:
            corr, _ = pearsonr(feature_values, sample_losses)
            # We care about the magnitude of correlation
            if np.isnan(corr):
                corr = 0.0
        correlations.append((i, corr))

    # Sort by absolute correlation (descending)
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 features correlated with prediction error (Validation Set):")
    for idx, corr in correlations[:5]:
        print(f"Feature Index {idx}: Correlation = {corr:.4f}")

    # 6. Submission Logic
    # Strict threshold from requirements
    THRESHOLD = 1.4705366241156435e-08

    if val_loss < THRESHOLD:
        print(
            f"\nValidation metric ({val_loss}) is lower than threshold ({THRESHOLD})."
        )
        print("Generating predictions for test set...")

        # Predict on test set
        y_test_pred = model.predict_proba(X_test)

        # Save submission
        save_submission(ids_test, y_test_pred, model.classes_, filename=SUBMISSION_PATH)
    else:
        print(
            f"\nValidation metric ({val_loss}) is NOT lower than threshold ({THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
