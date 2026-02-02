import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from library import config, utils, preprocessing, model


def main():
    # 1. Initialization and Reproducibility
    utils.set_seed(config.SEED)
    print("Environment initialized and random seed set.")

    # 2. Data Loading
    # Utilizes the pipeline defined in library.features via library.preprocessing
    # This handles feature extraction (Geometric + Tabular), Sanitization, and Scaling.
    print("Loading data...")
    try:
        X_train, y_train, X_val, y_val, X_test, test_ids, classes = (
            preprocessing.load_data(load_cached_data=True, debug=False)
        )
    except Exception as e:
        print(f"Failed to load data: {e}")
        sys.exit(1)

    print(f"Data Loaded successfully.")
    print(f"Training Set: {X_train.shape}")
    print(f"Validation Set: {X_val.shape}")
    print(f"Test Set: {X_test.shape}")
    print(f"Number of Classes: {len(classes)}")

    # 3. Model Training
    # The OASDiscriminant implements the strategy:
    # - Empirical Means/Priors
    # - OAS Covariance Estimation on Residuals
    # - Linear Decision Boundary Derivation (W, b)
    # - Float64 Precision
    print("Initializing and training OASDiscriminant...")
    clf = model.OASDiscriminant()
    clf.fit(X_train, y_train)
    print("Model trained successfully.")

    # 4. Validation
    print("Performing validation inference...")
    val_probs = clf.predict_proba(X_val)

    # Compute Multi-class Log Loss
    # Cite solution_lesson_node_00184: Pass raw softmax probabilities directly to log_loss
    # to avoid metric degradation caused by "Clip-then-Normalize" artifacts.
    val_loss = log_loss(y_val, val_probs, labels=np.arange(len(classes)))

    # REQUIRED OUTPUT: Print the final validation metric
    print(f"Final Validation Metric: {val_loss}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate error magnitude: 1.0 - probability assigned to the true class
    # y_val contains the integer indices of the true classes
    true_class_indices = y_val
    rows = np.arange(len(y_val))

    # Extract prob of true class
    true_probs = val_probs[rows, true_class_indices]
    error_magnitude = 1.0 - true_probs

    # Calculate correlation between error magnitude and each feature
    # X_val is (n_samples, n_features)
    n_features = X_val.shape[1]
    correlations = []

    for i in range(n_features):
        feature_vals = X_val[:, i]
        # Check for constant features to avoid division by zero in correlation
        if np.std(feature_vals) < 1e-12:
            corr = 0.0
        else:
            corr = np.corrcoef(feature_vals, error_magnitude)[0, 1]
        correlations.append(corr)

    correlations = np.array(correlations)

    # Identify top 5 features most correlated (positively or negatively) with error
    top_indices = np.argsort(np.abs(correlations))[::-1][:5]

    print("Top 5 Features Correlated with Error Magnitude:")
    for idx in top_indices:
        print(f"Feature Index {idx}: Correlation = {correlations[idx]:.6f}")

    # 6. Submission Generation
    # Threshold defined in the prompt
    THRESHOLD = 3.058881515561734e-14

    if val_loss < THRESHOLD:
        print(f"\nValidation metric ({val_loss}) meets the threshold ({THRESHOLD}).")
        print("Generating submission...")

        # Inference on Test Set
        test_probs = clf.predict_proba(X_test)

        # Clip probabilities
        eps = 1e-15
        test_probs_clipped = np.clip(test_probs, eps, 1 - eps)

        # Construct Submission DataFrame
        # Columns must be 'id' followed by species names
        submission_df = pd.DataFrame(test_probs_clipped, columns=classes)
        submission_df.insert(0, "id", test_ids)

        # Save to file
        submission_path = "./submission/submission.csv"
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(
            f"\nValidation metric ({val_loss}) does NOT meet the threshold ({THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
