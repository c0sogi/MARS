import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelBinarizer

# Import from provided library files
from library.config import Config
from library.utils import set_seed, ensure_float64
from library.data_loader import DataLoader
from library.model import OASDiscriminant


def main():
    # 1. Setup and Reproducibility
    set_seed(Config.SEED)
    print("Runfile: Initialized environment and seeds.")

    # 2. Data Loading
    # The DataLoader handles feature extraction (morphology), merging, and robust preprocessing
    # It returns float64 numpy arrays and pandas Series for targets/ids
    print("Runfile: Loading data...")
    loader = DataLoader()
    X_train, y_train, X_val, y_val, X_test, test_ids = loader.load_data(
        load_cached_data=True
    )

    print(
        f"Runfile: Data loaded. Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}"
    )

    # 3. Model Training
    # Initialize the OAS Discriminant with the strategy-defined parameters
    print("Runfile: Training OASDiscriminant model...")
    model = OASDiscriminant(assume_centered=Config.OAS_ASSUME_CENTERED)

    # Fit on training data
    # The model handles class prior and mean computation, then estimates covariance via OAS
    model.fit(X_train, y_train)
    print("Runfile: Model training complete.")

    # 4. Validation Inference
    print("Runfile: Running validation inference...")
    # Predict probabilities (already float64 and softmaxed)
    val_probs = model.predict_proba(X_val)

    # 5. Metric Calculation
    # The metric is Multi-class log loss.
    # We must clip probabilities as per the task description: max(min(p, 1-1e-15), 1e-15)
    # Note: sklearn log_loss does this by default with eps=1e-15, but we make it explicit for correctness.
    eps = 1e-15
    val_probs_clipped = np.clip(val_probs, eps, 1 - eps)

    # Calculate Log Loss
    # We need to ensure y_val is properly formatted for log_loss
    # log_loss handles string labels if we provide the labels parameter,
    # but using a binarizer ensures we match the probability columns order.
    lb = LabelBinarizer()
    lb.fit(model.classes_)
    y_val_bin = lb.transform(y_val)

    # Calculate metric
    final_metric = log_loss(y_val_bin, val_probs_clipped)

    # Print metric in the required format with full precision
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\nRunfile: Performing failure analysis...")
    # Calculate per-sample log loss (Cross Entropy)
    # element-wise multiplication of true class binary vector and log(probs)
    # sum over classes, then negate.
    # Since y_val_bin is 1-hot, this picks the log prob of the true class.
    per_sample_loss = -np.sum(y_val_bin * np.log(val_probs_clipped), axis=1)

    # Correlate error magnitude with input features
    # X_val is (n_samples, n_features), per_sample_loss is (n_samples,)
    # We compute correlation for each feature column
    correlations = []
    n_features = X_val.shape[1]

    # Avoid division by zero in correlation if a feature has 0 variance in validation set
    loss_std = np.std(per_sample_loss)

    if loss_std > 0:
        for i in range(n_features):
            feature_col = X_val[:, i]
            feat_std = np.std(feature_col)
            if feat_std > 0:
                corr = np.corrcoef(feature_col, per_sample_loss)[0, 1]
                correlations.append((i, corr))
            else:
                correlations.append((i, 0.0))
    else:
        correlations = [(i, 0.0) for i in range(n_features)]

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 features correlated with error magnitude:")
    # We don't have feature names easily accessible here as X is numpy array,
    # but we can refer to indices.
    for idx, corr in correlations[:5]:
        print(f"  Feature Index {idx}: Correlation = {corr:.4f}")

    # 7. Submission Generation
    # Threshold check
    threshold = 3.3382359570696616e-14

    if final_metric < threshold:
        print(
            f"\nRunfile: Metric {final_metric} meets threshold {threshold}. Generating submission..."
        )

        # Predict on test set
        test_probs = model.predict_proba(X_test)

        # Clip probabilities for submission consistency
        test_probs_clipped = np.clip(test_probs, eps, 1 - eps)

        # Create DataFrame
        # Columns must be the species names in order
        submission_df = pd.DataFrame(test_probs_clipped, columns=model.classes_)

        # Insert ID column at the beginning
        submission_df.insert(0, "id", test_ids.values)

        # Save to file
        submission_path = Config.SUBMISSION_FILE_PATH
        submission_df.to_csv(submission_path, index=False)
        print(f"Runfile: Submission saved to {submission_path}")

    else:
        print(
            f"\nRunfile: Metric {final_metric} did not meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
