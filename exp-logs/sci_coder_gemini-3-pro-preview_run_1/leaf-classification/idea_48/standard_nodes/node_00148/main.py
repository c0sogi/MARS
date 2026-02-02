import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from library.config import Config
from library.data_loader import load_and_process_data
from library.oas_lda import OASDiscriminant


def run_pipeline():
    # 1. Setup and Configuration
    print("Initializing Pipeline...")
    Config.setup()

    # 2. Load Data
    # Utilizing cached data for speed as per instructions
    print("Loading preprocessed data...")
    try:
        # Cite debug_lesson_2: Invalidate Stale Cache Artifacts When Switching Execution Contexts
        X_train, y_train, X_val, y_val, X_test, test_ids, classes = (
            load_and_process_data(load_cached_data=False, debug=False)
        )
    except Exception as e:
        print(f"Critical Error during data loading: {e}")
        sys.exit(1)

    print(f"Data Loaded Successfully.")
    print(f"Train shape: {X_train.shape}")
    print(f"Val shape:   {X_val.shape}")
    print(f"Test shape:  {X_test.shape}")

    # 3. Model Training
    # OAS Discriminant is a closed-form solution, fitting is fast.
    print("Training OASDiscriminant Model...")
    model = OASDiscriminant()
    model.fit(X_train, y_train)
    print("Model training complete.")

    # 4. Validation Inference & Metric Calculation
    print("Evaluating on Validation Set...")
    val_probs = model.predict_proba(X_val)

    # Calculate Multi-class Log Loss
    # Scikit-learn's log_loss handles the clipping (eps=1e-15) internally, matching the task spec.
    val_loss = log_loss(y_val, val_probs, labels=np.arange(len(classes)))

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_loss}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate per-sample loss for correlation analysis
    # Get the probability assigned to the true class
    rows = np.arange(len(y_val))
    true_class_probs = val_probs[rows, y_val]
    # Clip probabilities to prevent log(0)
    true_class_probs = np.clip(true_class_probs, 1e-15, 1.0 - 1e-15)
    sample_losses = -np.log(true_class_probs)

    # Compute correlation between features and error magnitude
    # We iterate through features by index since we are working with numpy arrays
    n_features = X_val.shape[1]
    correlations = []

    # Calculate correlations efficiently
    # Center the loss
    loss_centered = sample_losses - np.mean(sample_losses)
    loss_std = np.std(sample_losses)

    if loss_std > 0:
        for i in range(n_features):
            feat_col = X_val[:, i]
            feat_std = np.std(feat_col)
            if feat_std > 0:
                # Pearson correlation
                corr = np.mean((feat_col - np.mean(feat_col)) * loss_centered) / (
                    feat_std * loss_std
                )
                correlations.append((i, corr))
            else:
                correlations.append((i, 0.0))
    else:
        print(
            "Loss variance is zero (all predictions identical quality). Skipping correlation analysis."
        )
        correlations = [(i, 0.0) for i in range(n_features)]

    # Sort by absolute correlation strength
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Prediction Error:")
    for idx, corr in correlations[:5]:
        print(f"Feature Index {idx}: Correlation = {corr:.4f}")

    # 6. Submission
    # The task specifies a threshold.
    THRESHOLD = 3.3382359570696616e-14

    # Check condition
    if val_loss < THRESHOLD:
        print(
            f"\nValidation metric ({val_loss}) meets the strict threshold ({THRESHOLD})."
        )
    else:
        print(
            f"\nValidation metric ({val_loss}) does not meet the strict threshold ({THRESHOLD})."
        )
        print("Proceeding with submission generation to satisfy file requirements.")

    print("Generating Test Predictions...")
    test_probs = model.predict_proba(X_test)

    print("Formatting Submission...")
    # Construct DataFrame
    submission_df = pd.DataFrame(test_probs, columns=classes)
    # Insert ID column at the beginning
    submission_df.insert(0, "id", test_ids)

    # Ensure directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)

    # Save
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")


if __name__ == "__main__":
    run_pipeline()
