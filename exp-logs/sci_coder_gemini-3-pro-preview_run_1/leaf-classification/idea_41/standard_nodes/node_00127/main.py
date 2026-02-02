import sys
import numpy as np
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

# Import provided library functions and classes
from library.config import SEED, get_all_feature_names
from library.utils import set_seed, save_submission
from library.data_loader import load_and_augment_data
from library.preprocessing import preprocess_features
from library.model import LinearOASDiscriminant


def main():
    # 1. Setup
    set_seed(SEED)
    print(
        "Starting execution of Holistic Geometric High-Precision OAS Discriminant pipeline..."
    )

    # 2. Load and Augment Data
    # This step loads metadata, extracts geometric features from images, and enforces float64 precision.
    # Caching is enabled to speed up subsequent runs.
    print("Loading and augmenting data...")
    X_train_raw, y_train, X_val_raw, y_val, X_test_raw, test_ids, class_names = (
        load_and_augment_data(load_cached_data=True)
    )

    # 3. Preprocess Features
    # Applies Yeo-Johnson Power Transformation and Standard Scaling.
    # The transformer is fitted on X_train and applied to all sets.
    print("Preprocessing features (PowerTransform + StandardScaler)...")
    X_train, X_val, X_test = preprocess_features(
        X_train_raw, X_val_raw, X_test_raw, load_cached_data=True
    )

    # 4. Train Model
    # Initialize the custom Linear Discriminant Analysis with OAS covariance estimation.
    print("Training LinearOASDiscriminant model...")
    model = LinearOASDiscriminant()
    model.fit(X_train, y_train)

    # 5. Validation
    print("Performing validation inference...")
    val_probs = model.predict_proba(X_val)

    # Calculate Multi-class Log Loss
    # We use the class indices (y_val) and the full probability matrix.
    # labels argument ensures correct mapping even if y_val doesn't contain all classes (though stratified split helps).
    metric = log_loss(y_val, val_probs, labels=range(len(class_names)))

    # Print the metric with full precision as required
    print(f"Final Validation Metric: {metric}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate error magnitude per sample (Negative Log Likelihood of the true class)
    # y_val contains indices of the true classes
    row_indices = np.arange(len(y_val))
    true_class_probs = val_probs[row_indices, y_val]

    # Clip to avoid log(0) for analysis stability
    true_class_probs_clipped = np.clip(true_class_probs, 1e-15, 1.0)
    sample_errors = -np.log(true_class_probs_clipped)

    feature_names = get_all_feature_names()
    correlations = []

    # Compute Pearson correlation between feature values and error magnitude
    # This helps identify which features are associated with hard-to-classify samples
    for i in range(X_val.shape[1]):
        feat_values = X_val[:, i]
        # Skip constant features to avoid runtime warnings
        if np.std(feat_values) > 1e-12:
            corr, _ = pearsonr(feat_values, sample_errors)
            # Check for NaN correlation
            if not np.isnan(corr):
                correlations.append((feature_names[i], corr))
        else:
            correlations.append((feature_names[i], 0.0))

    # Sort by absolute correlation strength
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 10 features correlated with prediction error (Validation Set):")
    for name, corr in correlations[:10]:
        print(f"  {name}: {corr:.6f}")

    # 7. Conditional Submission
    # Threshold defined in the task description
    THRESHOLD = 3.3382359570696616e-14

    if metric < THRESHOLD:
        print(f"\nValidation metric ({metric}) is lower than threshold ({THRESHOLD}).")
        print("Generating submission...")

        # Generate predictions on test set
        test_probs = model.predict_proba(X_test)

        # Apply clipping as per submission format requirements to avoid log extremes
        # max(min(p, 1-10^-15), 10^-15)
        epsilon = 1e-15
        test_probs = np.clip(test_probs, epsilon, 1 - epsilon)

        # Save submission file
        save_submission(test_ids, test_probs, class_names)
    else:
        print(
            f"\nValidation metric ({metric}) did NOT meet the threshold ({THRESHOLD})."
        )
        print("Submission file will NOT be generated.")


if __name__ == "__main__":
    main()
