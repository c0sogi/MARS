import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

# Import provided library components
from library.config import Config, set_seed
from library.data_loader import LeafDataManager
from library.preprocessor import SanitizedTransformPipeline
from library.oas_discriminant import OASLinearDiscriminant


def main():
    # 1. Initialization
    set_seed(Config.SEED)

    # 2. Data Loading
    print("Initializing Data Manager...")
    data_manager = LeafDataManager()

    print("Loading Training Data...")
    X_train, y_train, ids_train, feat_names = data_manager.load_data(
        "train", load_cached_data=True
    )

    print("Loading Validation Data...")
    X_val, y_val, ids_val, _ = data_manager.load_data("val", load_cached_data=True)

    print("Loading Test Data...")
    X_test, _, ids_test, _ = data_manager.load_data("test", load_cached_data=True)

    # 3. Preprocessing
    print("Fitting Preprocessing Pipeline...")
    pipeline = SanitizedTransformPipeline()

    # Fit on Train
    X_train_proc = pipeline.fit_transform(X_train)

    # Transform Val and Test
    X_val_proc = pipeline.transform(X_val)
    X_test_proc = pipeline.transform(X_test)

    # 4. Model Training
    print("Training OAS Linear Discriminant...")
    model = OASLinearDiscriminant()
    model.fit(X_train_proc, y_train)

    # 5. Validation & Metric
    print("Performing Validation...")
    val_probs = model.predict_proba(X_val_proc)

    # Calculate Multi-class Log Loss
    # We pass model.classes_ to ensure the columns of val_probs match the labels
    metric = log_loss(y_val, val_probs, labels=model.classes_)

    print(f"Final Validation Metric: {metric:.16f}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate error per sample: -log(p_true)
    class_to_idx = {cls: i for i, cls in enumerate(model.classes_)}
    y_val_indices = np.array([class_to_idx[y] for y in y_val])

    # Extract probability assigned to the true class
    # Clip to match the metric calculation stability
    eps = 1e-15
    val_probs_clipped = np.clip(val_probs, eps, 1 - eps)
    true_class_probs = val_probs_clipped[np.arange(len(y_val)), y_val_indices]

    errors = -np.log(true_class_probs)

    # Correlate errors with raw feature values
    correlations = []
    for i in range(X_val.shape[1]):
        feature_vals = X_val[:, i]
        # Skip constant features to avoid warnings
        if np.std(feature_vals) > 1e-12:
            corr, _ = pearsonr(feature_vals, errors)
            if np.isfinite(corr):
                correlations.append((feat_names[i], corr))

    # Sort by absolute correlation magnitude
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features Correlated with Error:")
    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.4f}")

    # 7. Submission Generation
    THRESHOLD = 3.058881515561734e-14

    if metric < THRESHOLD:
        print(f"\nValidation metric ({metric:.4e}) meets threshold ({THRESHOLD:.4e}).")
        print("Generating submission...")

        test_probs = model.predict_proba(X_test_proc)

        # Construct Submission DataFrame
        submission_df = pd.DataFrame(test_probs, columns=model.classes_)
        submission_df.insert(0, "id", ids_test)

        # Ensure directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        # Save
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation metric ({metric:.4e}) does NOT meet threshold ({THRESHOLD:.4e})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
