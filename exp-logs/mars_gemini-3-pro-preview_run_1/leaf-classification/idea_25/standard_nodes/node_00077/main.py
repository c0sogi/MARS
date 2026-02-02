import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

# Import provided library modules
from library import config
from library import data_loader
from library import preprocessing
from library import model


def main():
    # Set random seeds for reproducibility
    np.random.seed(config.SEED)

    print("=== Starting Runfile Execution ===")

    # -------------------------------------------------------------------------
    # 1. Data Loading
    # -------------------------------------------------------------------------
    print("Loading dataset...")
    # Load data using the cached loader
    # This returns raw features (float64) and labels/IDs
    X_train, y_train, X_val, y_val, X_test, test_ids = data_loader.load_dataset(
        load_cached_data=True
    )

    print(f"Train set: {X_train.shape} samples")
    print(f"Val set:   {X_val.shape} samples")
    print(f"Test set:  {X_test.shape} samples")

    # -------------------------------------------------------------------------
    # 2. Preprocessing
    # -------------------------------------------------------------------------
    print("Preprocessing features (Yeo-Johnson + Standard Scaling in float64)...")
    # get_transformed_data handles fitting on train and transforming all sets
    # It also manages caching of the transformed numpy arrays
    X_train_trans, X_val_trans, X_test_trans = preprocessing.get_transformed_data(
        X_train, X_val, X_test, load_cached_data=True
    )

    # -------------------------------------------------------------------------
    # 3. Model Training
    # -------------------------------------------------------------------------
    print("Initializing and training CalibratedOASDiscriminant...")
    # The model implements the Hybrid Generative-Discriminative strategy
    clf = model.CalibratedOASDiscriminant()

    # Fit the model
    # The model handles label encoding internally using LabelEncoder
    clf.fit(X_train_trans, y_train)
    print("Training complete.")

    # -------------------------------------------------------------------------
    # 4. Validation & Metric Calculation
    # -------------------------------------------------------------------------
    print("Performing validation inference...")
    # Predict probabilities on validation set
    # The model's predict_proba ensures float64 precision and clipping to [1e-15, 1-1e-15]
    val_probs = clf.predict_proba(X_val_trans)

    # Calculate Multi-class Log Loss
    # We pass the classes explicitly to ensure correct mapping between string labels and probability columns
    final_metric = log_loss(y_val, val_probs, labels=clf.classes_)

    # Print the required metric string strictly
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 5. Failure Analysis
    # -------------------------------------------------------------------------
    print("\n--- Failure Analysis ---")

    # Map string labels to integer indices to extract the probability of the true class
    class_to_idx = {cls: i for i, cls in enumerate(clf.classes_)}
    y_val_indices = np.array([class_to_idx[label] for label in y_val])

    # Extract probability assigned to the true class for each sample
    # val_probs is (N_val, N_classes)
    true_class_probs = val_probs[np.arange(len(y_val)), y_val_indices]

    # Calculate individual log loss for each sample: -log(p_true)
    sample_losses = -np.log(true_class_probs)

    # Calculate correlation between each feature and the error magnitude (sample_losses)
    n_features = X_val_trans.shape[1]
    correlations = []

    # Calculate standard deviation of losses once
    loss_std = np.std(sample_losses)

    if loss_std > 1e-15:
        for i in range(n_features):
            feature_vals = X_val_trans[:, i]
            feat_std = np.std(feature_vals)

            if feat_std > 1e-15:
                # Pearson correlation
                covariance = np.mean(
                    (feature_vals - np.mean(feature_vals))
                    * (sample_losses - np.mean(sample_losses))
                )
                corr = covariance / (feat_std * loss_std)
                correlations.append((config.FEATURE_COLUMNS[i], corr))
            else:
                correlations.append((config.FEATURE_COLUMNS[i], 0.0))
    else:
        print("Validation loss variance is negligible. Skipping feature correlation.")
        correlations = [(col, 0.0) for col in config.FEATURE_COLUMNS]

    # Sort by absolute correlation strength
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Prediction Error:")
    for feat_name, corr_val in correlations[:5]:
        print(f"  {feat_name}: {corr_val:.4f}")

    # -------------------------------------------------------------------------
    # 6. Submission Generation
    # -------------------------------------------------------------------------
    # Threshold defined in the prompt
    THRESHOLD = 1.2136771218566717e-09

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is below threshold ({THRESHOLD}). Generating submission..."
        )

        # Predict on test set
        test_probs = clf.predict_proba(X_test_trans)

        # Create submission DataFrame
        # Columns must be: id, class_1, class_2, ...
        submission_df = pd.DataFrame(test_probs, columns=clf.classes_)

        # Insert ID column at the beginning
        submission_df.insert(0, config.ID_COLUMN, test_ids)

        # Ensure submission directory exists
        os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

        # Save to CSV
        submission_df.to_csv(config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {config.SUBMISSION_FILE}")

    else:
        print(
            f"\nMetric ({final_metric}) did NOT meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
