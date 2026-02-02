import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from library.config import RANDOM_SEED, SUBMISSION_PATH
from library.transformations import get_transformed_data
from library.oas_model import OASLinearDiscriminant, create_submission_file


def run_pipeline():
    # 1. Set Seed for reproducibility
    np.random.seed(RANDOM_SEED)

    # 2. Load and Transform Data
    # This leverages the library to load raw data, extract geometric features,
    # and apply the Iterative Gaussianization pipeline (YJ -> Whitening PCA -> YJ).
    print("Loading and transforming data...")
    (train_data, val_data, test_data) = get_transformed_data(load_cached_data=True)

    X_train, y_train, ids_train = train_data
    X_val, y_val, ids_val = val_data
    X_test, ids_test = test_data

    # 3. Train Model
    # The OASLinearDiscriminant uses float64 precision and an analytical solver.
    print("Training OAS Linear Discriminant...")
    model = OASLinearDiscriminant()
    model.fit(X_train, y_train)

    # 4. Validation Inference
    print("Running validation inference...")
    # Predict probabilities (Softmax output)
    probs_val = model.predict_proba(X_val)

    # 5. Calculate Metric (Multi-class Log Loss)
    # We strictly follow the scoring protocol: Rescale -> Clip -> Score

    # A. Rescale (Normalize rows to sum to 1)
    # Note: Softmax already sums to 1, but we enforce it to handle any float drift
    row_sums = probs_val.sum(axis=1)
    probs_val_norm = probs_val / row_sums[:, np.newaxis]

    # B. Clip to avoid log(0) extremes
    epsilon = 1e-15
    probs_val_clipped = np.clip(probs_val_norm, epsilon, 1.0 - epsilon)

    # C. Score
    # Map validation labels to column indices
    le = LabelEncoder()
    le.fit(model.classes_)
    y_val_indices = le.transform(y_val)

    # Extract probabilities corresponding to the true class
    true_class_probs = probs_val_clipped[np.arange(len(y_val)), y_val_indices]

    # Compute negative log likelihood
    log_vals = np.log(true_class_probs)
    final_metric = -np.mean(log_vals)

    # Print the metric with full precision
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("Performing failure analysis...")
    # Error magnitude is defined as the negative log probability of the true class (Cross Entropy)
    sample_losses = -np.log(true_class_probs)

    # Calculate Pearson correlation between error magnitude and each transformed feature
    n_features = X_val.shape[1]
    correlations = []

    for i in range(n_features):
        feature_vals = X_val[:, i]
        # Check for constant features to avoid division by zero
        if np.std(feature_vals) == 0 or np.std(sample_losses) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(feature_vals, sample_losses)[0, 1]
        correlations.append((i, corr))

    # Sort by absolute correlation strength
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 features correlated with error magnitude:")
    for idx, corr in correlations[:5]:
        print(f"Feature {idx}: Correlation = {corr:.4f}")

    # 7. Submission Generation
    # Strict threshold check as per requirements
    THRESHOLD = 3.3382359570696616e-14

    if final_metric < THRESHOLD:
        print("Metric below threshold. Generating submission...")
        create_submission_file(model, X_test, ids_test, SUBMISSION_PATH)
    else:
        print(f"Metric {final_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    run_pipeline()
