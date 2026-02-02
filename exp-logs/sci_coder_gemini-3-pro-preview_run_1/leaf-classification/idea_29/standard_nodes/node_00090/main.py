import sys
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import provided library modules
from library import config
from library import data_processor
from library import model
from library import evaluation


def run():
    # 1. Set fixed random seeds for reproducibility
    np.random.seed(config.SEED)

    # 2. Load Data
    # Utilizing the preprocessed data from the working directory via load_cached_data=True
    print("Initializing Data Processor...")
    processor = data_processor.LeafDataProcessor()
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = processor.load_data(
        load_cached_data=True
    )

    # 3. Model Training
    # The CholeskyOASClassifier is a direct linear solver and fits very quickly on CPU.
    print("Initializing and training CholeskyOASClassifier...")
    clf = model.CholeskyOASClassifier()
    clf.fit(X_train, y_train)

    # 4. Validation
    print("Performing validation...")
    # Predict probabilities on the validation set
    val_probs = clf.predict_proba(X_val)

    # Compute the multi-class log loss using the evaluation utility
    # This handles clipping and consistent metric calculation
    val_loss = evaluation.compute_log_loss(y_val, val_probs)

    # Print the validation metric in the strictly required format
    print(f"Final Validation Metric: {val_loss}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate error magnitude per sample: Negative Log Likelihood of the true class
    # We clip probabilities to match the metric's stability constraints
    eps = 1e-15
    val_probs_clipped = np.clip(val_probs, eps, 1 - eps)

    # Extract the predicted probability for the true class label for each sample
    n_samples = len(y_val)
    # y_val contains integer encoded labels corresponding to indices in `classes`
    true_class_probs = val_probs_clipped[np.arange(n_samples), y_val]

    # Error magnitude is high when true class probability is low
    error_magnitude = -np.log(true_class_probs)

    # Compute Pearson correlation between each feature and the error magnitude
    feature_names = config.FEATURE_COLS
    correlations = []

    # Iterate through features to find which are most associated with model errors
    for idx, feature in enumerate(feature_names):
        if idx < X_val.shape[1]:
            feat_values = X_val[:, idx]

            # Calculate correlation only if feature has variance
            if np.std(feat_values) > 0:
                corr, _ = pearsonr(feat_values, error_magnitude)
                # Handle potential NaNs
                if np.isnan(corr):
                    corr = 0.0
            else:
                corr = 0.0
            correlations.append((feature, corr))

    # Sort features by the absolute value of their correlation with error
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 10 Features correlated with Error Magnitude:")
    for name, corr in correlations[:10]:
        print(f"{name}: {corr:.6f}")

    # 6. Submission Generation
    # The task specifies a strict threshold for generating the submission
    THRESHOLD = 1.2136771218566717e-09

    if val_loss < THRESHOLD:
        print(
            f"\nValidation metric ({val_loss}) is strictly lower than threshold ({THRESHOLD})."
        )
        print("Generating submission for test set...")

        # Generate predictions for the test set
        test_probs = clf.predict_proba(X_test)

        # Save submission using the evaluation utility
        evaluation.save_submission(test_ids, test_probs, classes)
    else:
        print(
            f"\nValidation metric ({val_loss}) is NOT lower than threshold ({THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    run()
