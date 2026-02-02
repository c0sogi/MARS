import sys
import os
import numpy as np
import random
import warnings

# Import provided library modules
from library import config
from library import utils
from library import preprocessing
from library import model

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seeds(seed):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Set torch seeds if available, though primarily using sklearn here
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def main():
    # 1. Setup
    set_seeds(config.SEED)

    # 2. Load and Preprocess Data
    # The preprocessing module handles loading metadata from ./metadata,
    # extracting features, and applying PowerTransformer + StandardScaler.
    print("Loading and preprocessing data...")
    data = preprocessing.get_preprocessed_data(load_cached_data=True)
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = data

    print(f"Training Data Shape: {X_train.shape}")
    print(f"Validation Data Shape: {X_val.shape}")
    print(f"Test Data Shape: {X_test.shape}")

    # 3. Model Training
    print("Initializing and training LDA...")
    # Initialize the single LDA model (Cite solution_lesson_node_00008, solution_lesson_node_00011)
    clf = model.get_model()

    # Fit the model
    clf.fit(X_train, y_train)

    # 4. Validation
    print("Performing validation inference...")
    # Predict probabilities on the validation set
    y_pred_val = clf.predict_proba(X_val)

    # Calculate Log Loss using the provided utility which handles clipping/normalization
    val_metric = utils.calculate_log_loss(y_val, y_pred_val)

    # REQUIRED: Print the final validation metric in the specific format
    print(f"Final Validation Metric: {val_metric}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate error magnitude: 1.0 - Probability assigned to the true class
    # y_val contains the true class indices
    # y_pred_val contains the predicted probabilities

    # Extract probability of the true class for each sample
    row_indices = np.arange(len(y_val))
    prob_true_class = y_pred_val[row_indices, y_val]

    # Error magnitude (higher is worse)
    error_magnitude = 1.0 - prob_true_class

    # Calculate Pearson correlation between error magnitude and input features
    n_features = X_val.shape[1]
    correlations = []

    # Check if there is variance in error to avoid division by zero
    if np.std(error_magnitude) > 0:
        for i in range(n_features):
            feature_vals = X_val[:, i]
            # Calculate correlation if feature has variance
            if np.std(feature_vals) > 0:
                corr = np.corrcoef(feature_vals, error_magnitude)[0, 1]
                # Handle NaN result
                if np.isnan(corr):
                    corr = 0.0
            else:
                corr = 0.0
            correlations.append((i, corr))
    else:
        print("Model error is constant (likely 0.0). No correlations to report.")
        correlations = [(i, 0.0) for i in range(n_features)]

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 features correlated with error magnitude:")
    for idx, corr in correlations[:5]:
        print(f"Feature {idx}: Correlation = {corr:.6f}")

    # 6. Submission
    # Threshold defined in the task
    threshold = 1.4705545687989736e-08

    if val_metric < threshold:
        print(f"\nValidation metric ({val_metric}) meets threshold ({threshold}).")
        print("Generating predictions for test set...")

        # Predict on test set
        y_pred_test = clf.predict_proba(X_test)

        # Save submission
        utils.save_submission(
            test_ids, y_pred_test, classes, output_path=config.SUBMISSION_CSV
        )
    else:
        print(
            f"\nValidation metric ({val_metric}) does not meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
