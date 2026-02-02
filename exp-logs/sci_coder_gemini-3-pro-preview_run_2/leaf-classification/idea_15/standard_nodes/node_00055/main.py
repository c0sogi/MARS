import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

# Import from provided library files
from library.utils import set_seed, create_submission_file
from library.model_factory import (
    build_linear_branch,
    build_generative_branch,
)
from library.engine import train_ensemble, predict_ensemble
from library.data_loader import load_dataset, extract_geometric_features


def run():
    # 1. Set Seed for Reproducibility
    set_seed(42)

    # 2. Validation Phase
    # We manually load the split data to perform strict validation on the hold-out set
    print("Loading validation split metadata...")
    train_path = "./metadata/train.csv"
    val_path = "./metadata/val.csv"

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError("Metadata files not found.")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    # Extract features (exclude non-feature columns)
    exclude_cols = ["id", "species", "image_path"]
    feature_cols = sorted([c for c in df_train.columns if c not in exclude_cols])

    X_train_split = df_train[feature_cols].values.astype(np.float32)
    X_val_split = df_val[feature_cols].values.astype(np.float32)

    # Extract and append geometric features for the validation splits
    print("Extracting geometric features for validation splits...")
    geo_train = extract_geometric_features(df_train["image_path"].values)
    geo_val = extract_geometric_features(df_val["image_path"].values)

    X_train_split = np.hstack([X_train_split, geo_train])
    X_val_split = np.hstack([X_val_split, geo_val])

    # Update feature columns list for failure analysis
    feature_cols.extend(["aspect_ratio", "extent", "solidity", "eccentricity"])

    # Encode targets
    le = LabelEncoder()
    y_train_split = le.fit_transform(df_train["species"])
    # Transform val labels; assumes all val classes exist in train (verified in metadata analysis)
    y_val_split = le.transform(df_val["species"])

    print(
        f"Training on {X_train_split.shape[0]} samples, Validating on {X_val_split.shape[0]} samples."
    )

    # Train Ensemble on the Training Split
    models_split = train_ensemble(X_train_split, y_train_split, random_state=42)

    # Inference on Validation Set
    val_probs = predict_ensemble(models_split, X_val_split)

    # Compute Metric (Log Loss)
    # Clip probabilities to avoid log(0) extremes, consistent with competition metric rules
    epsilon = 1e-15
    val_probs_clipped = np.clip(val_probs, epsilon, 1 - epsilon)
    val_metric = log_loss(y_val_split, val_probs_clipped)

    # Print the required validation metric
    print(f"Final Validation Metric: {val_metric}")

    # 3. Failure Analysis
    print("Performing Failure Analysis...")
    # Calculate per-sample log loss: -log(p_true_class)
    n_samples = len(y_val_split)
    # Advanced indexing to select probability of the true class for each sample
    p_true = val_probs_clipped[np.arange(n_samples), y_val_split]
    sample_losses = -np.log(p_true)

    # Calculate correlation between feature values and error magnitude
    correlations = []
    for i, col_name in enumerate(feature_cols):
        feat_values = X_val_split[:, i]
        # Handle constant features to avoid division by zero in correlation
        if np.std(feat_values) < 1e-9:
            corr = 0.0
        else:
            # np.corrcoef returns correlation matrix
            corr = np.corrcoef(feat_values, sample_losses)[0, 1]

        if np.isnan(corr):
            corr = 0.0

        correlations.append((col_name, corr))

    # Sort by absolute correlation to find strongest relationships
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 features correlated with Error Magnitude (Failure Analysis):")
    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.6f}")

    # 4. Submission Phase
    # Strict threshold defined in the task
    THRESHOLD = 0.00870833951594525

    if val_metric < THRESHOLD:
        print(
            "Validation metric meets threshold. Proceeding to full training and submission..."
        )

        # Load full dataset (Train + Val) using the library function
        # We force reload (load_cached_data=False) to ensure we get the concatenated dataset
        # maximizing sample size for the final submission.
        X_full, y_full, X_test, test_ids, le_full = load_dataset(load_cached_data=False)

        # Retrain Ensemble on Full Data
        models_full = train_ensemble(X_full, y_full, random_state=42)

        # Inference on Test Set
        test_probs = predict_ensemble(models_full, X_test)

        # Create Submission File
        submission_path = "./submission/submission.csv"
        class_names = list(le_full.classes_)
        create_submission_file(test_ids, class_names, test_probs, submission_path)

    else:
        print(
            f"Validation metric {val_metric} is not lower than {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    run()
