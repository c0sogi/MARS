import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from library.config import set_seed, SUBMISSION_FILE_PATH, ID_COL
from library.data_loader import DataManager
from library.model import HighPrecisionOAS


def main():
    # 1. Setup and Reproducibility
    set_seed()

    # 2. Data Loading
    # We use the provided DataManager which handles:
    # - Loading metadata
    # - Extracting geometric features (Dual-Envelope Morphological Fusion)
    # - Merging with tabular features
    # - Preprocessing (Yeo-Johnson, Scaling, float64 conversion)
    # - Caching
    print("Initializing Data Pipeline...")
    dm = DataManager(load_cached_data=True)
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = dm.load_data()

    # 3. Model Training
    # The HighPrecisionOAS model is an analytical linear discriminant.
    # It does not require epochs or batches.
    print("Training HighPrecisionOAS Model...")
    model = HighPrecisionOAS()
    model.fit(X_train, y_train)

    # 4. Validation
    print("Performing Validation Inference...")
    # Predict probabilities on the validation set
    val_probs = model.predict_proba(X_val)

    # Calculate Multi-class Log Loss
    # sklearn.metrics.log_loss handles the clipping (eps=1e-15) internally,
    # matching the competition metric requirement.
    # We pass labels to ensure correct handling of all classes.
    val_metric = log_loss(y_val, val_probs, labels=range(len(classes)))

    # Print the exact metric as required
    print(f"Final Validation Metric: {val_metric}")

    # 5. Failure Analysis
    print("Performing Failure Analysis...")
    # Calculate per-sample error magnitude (Cross Entropy)
    # y_val contains the true class indices.
    # We extract the predicted probability for the true class.
    rows = np.arange(len(y_val))
    true_class_probs = val_probs[rows, y_val]

    # Clip probabilities for stable log calculation
    eps = 1e-15
    true_class_probs = np.clip(true_class_probs, eps, 1 - eps)
    error_magnitude = -np.log(true_class_probs)

    # Calculate correlation between error magnitude and input features
    n_features = X_val.shape[1]
    correlations = []

    for i in range(n_features):
        feature_vals = X_val[:, i]
        # Avoid correlation calculation if feature is constant
        if np.std(feature_vals) > 1e-12:
            corr = np.corrcoef(feature_vals, error_magnitude)[0, 1]
            correlations.append((i, corr))
        else:
            correlations.append((i, 0.0))

    # Sort by absolute correlation strength
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 features correlated with error magnitude:")
    for i in range(min(5, len(correlations))):
        idx, corr = correlations[i]
        print(f"Feature Index {idx}: Correlation {corr:.6f}")

    # 6. Submission Generation
    # Strict threshold check
    THRESHOLD = 3.3382359570696616e-14

    if val_metric < THRESHOLD:
        print(
            f"Validation metric {val_metric} meets threshold {THRESHOLD}. Generating submission..."
        )

        # Inference on Test Set
        test_probs = model.predict_proba(X_test)

        # Format Submission
        submission_df = pd.DataFrame(test_probs, columns=classes)
        submission_df.insert(0, ID_COL, test_ids)

        # Save to disk
        os.makedirs(os.path.dirname(SUBMISSION_FILE_PATH), exist_ok=True)
        submission_df.to_csv(SUBMISSION_FILE_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_FILE_PATH}")

    else:
        print(
            f"Validation metric {val_metric} does not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
