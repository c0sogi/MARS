import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

# Import provided library functions
from library.utils import set_seed
from library.data_loader import load_full_dataset
from library.ensemble_trainer import train_ensemble
from library.inference import predict_ensemble, generate_submission


def load_split_data_for_validation():
    """
    Manually loads train and validation sets from metadata to ensure a strict
    hold-out set for metric calculation and failure analysis.
    """
    # Paths
    train_path = "./metadata/train.csv"
    val_path = "./metadata/val.csv"

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(
            "Metadata files not found. Ensure ./metadata/train.csv and ./metadata/val.csv exist."
        )

    # Read CSVs
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    # Identify feature columns
    exclude_cols = {"id", "species", "image_path"}
    feature_cols = [c for c in df_train.columns if c not in exclude_cols]
    feature_cols.sort()

    # Extract features
    X_train = df_train[feature_cols].values.astype(np.float32)
    X_val = df_val[feature_cols].values.astype(np.float32)

    # Encode Targets
    # Fit LabelEncoder on all available species to ensure consistent mapping
    # between this validation run and the final full run.
    all_species = pd.concat([df_train["species"], df_val["species"]]).unique()
    all_species.sort()  # Ensure deterministic order

    le = LabelEncoder()
    le.fit(all_species)

    y_train = le.transform(df_train["species"])
    y_val = le.transform(df_val["species"])

    return X_train, y_train, X_val, y_val, feature_cols, le.classes_


def perform_failure_analysis(X_val, y_val, probs_val, feature_cols):
    """
    Analyzes prediction errors on the validation set.
    Calculates correlation between per-sample log loss and feature values.
    """
    print("\nPerforming Failure Analysis...")

    # Calculate Log Loss per sample: -log(p_true)
    # y_val contains indices of correct classes
    row_indices = np.arange(len(y_val))
    true_class_probs = probs_val[row_indices, y_val]

    # Clip to avoid log(0)
    epsilon = 1e-15
    true_class_probs = np.clip(true_class_probs, epsilon, 1 - epsilon)

    sample_losses = -np.log(true_class_probs)

    # Calculate correlation with each feature
    correlations = []
    for i, feature_name in enumerate(feature_cols):
        feature_values = X_val[:, i]

        # Skip constant features to avoid warnings
        if np.std(feature_values) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(sample_losses, feature_values)[0, 1]

        correlations.append((feature_name, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error (Log Loss):")
    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.4f}")


def main():
    # Ensure reproducibility
    set_seed(42)

    # =========================================================================
    # PHASE 1: VALIDATION
    # =========================================================================
    print("=== Validation Phase ===")

    # Load data
    X_train_split, y_train_split, X_val, y_val, feature_cols, classes_val = (
        load_split_data_for_validation()
    )

    # Train ensemble on the training split only
    print(f"Training ensemble on {len(X_train_split)} samples...")
    models_val = train_ensemble(X_train_split, y_train_split)

    # Predict on the hold-out validation set
    print(f"Evaluating on {len(X_val)} validation samples...")
    probs_val = predict_ensemble(models_val, X_val)

    # Compute Validation Metric
    # Provide labels to ensure log_loss handles all classes correctly
    val_metric = log_loss(y_val, probs_val, labels=np.arange(len(classes_val)))
    print(f"Final Validation Metric: {val_metric}")

    # Failure Analysis
    perform_failure_analysis(X_val, y_val, probs_val, feature_cols)

    # =========================================================================
    # PHASE 2: SUBMISSION
    # =========================================================================
    THRESHOLD = 0.010187299388940634

    if val_metric < THRESHOLD:
        print("\n=== Submission Phase ===")
        print(
            f"Validation metric {val_metric} < {THRESHOLD}. Proceeding to submission."
        )

        # Load full dataset (Train + Val) to maximize performance
        # Using the provided loader which handles caching and concatenation
        X_full, y_full, X_test, test_ids, classes_full = load_full_dataset(
            load_cached_data=True
        )

        # Retrain ensemble on the full dataset
        print("Retraining ensemble on full dataset...")
        models_full = train_ensemble(X_full, y_full)

        # Generate and save submission
        output_path = "./submission/submission.csv"
        generate_submission(
            models_full, X_test, test_ids, classes_full, output_path=output_path
        )

    else:
        print(f"\nValidation metric {val_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
