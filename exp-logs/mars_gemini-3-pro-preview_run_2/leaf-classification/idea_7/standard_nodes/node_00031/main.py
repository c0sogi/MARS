import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss

from library.config import (
    TRAIN_PATH,
    VAL_PATH,
    SUBMISSION_PATH,
    ID_COL,
    TARGET_COL,
    GENUS_COL,
    RANDOM_SEED,
)
from library.utils import set_seed
from library.hierarchical_engine import HybridEnsemble
from library.data_loader import load_and_process_data


def main():
    # 1. Setup
    set_seed(RANDOM_SEED)

    # 2. Manual Data Loading for Validation Phase
    # We need to explicitly load the split defined in metadata to compute the validation metric
    print("Loading metadata for validation split...")
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(VAL_PATH):
        raise FileNotFoundError("Metadata files not found.")

    df_train_split = pd.read_csv(TRAIN_PATH)
    df_val_split = pd.read_csv(VAL_PATH)

    # Identify feature columns (exclude metadata)
    # Note: 'species' is the target. 'id' and 'image_path' are metadata.
    non_feature_cols = [ID_COL, TARGET_COL, "image_path", GENUS_COL]
    feature_cols = [c for c in df_train_split.columns if c not in non_feature_cols]

    # Prepare Arrays
    X_train_raw = df_train_split[feature_cols].values
    y_train_species_raw = df_train_split[TARGET_COL].values

    X_val_raw = df_val_split[feature_cols].values
    y_val_species_raw = df_val_split[TARGET_COL].values

    # Encoding
    # We fit on the union of train and val to ensure all classes are handled and indices match
    all_species = np.concatenate([y_train_species_raw, y_val_species_raw])

    species_le = LabelEncoder()
    species_le.fit(all_species)
    y_train_species = species_le.transform(y_train_species_raw)
    y_val_species = species_le.transform(y_val_species_raw)

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    # 3. Train on Split
    print("Training HybridEnsemble on training split...")
    model = HybridEnsemble()
    model.fit(X_train, y_train_species)

    # 4. Validation Inference
    print("Evaluating on validation split...")
    val_probs = model.predict_proba(X_val)

    # Apply clipping as per metric definition
    val_probs = np.clip(val_probs, 1e-15, 1 - 1e-15)

    # Compute Metric
    # labels argument ensures that if a class is missing in y_val, it's still accounted for in probs columns
    metric = log_loss(y_val_species, val_probs, labels=range(len(species_le.classes_)))
    print(f"Final Validation Metric: {metric}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate per-sample log loss: -log(p_true)
    # Index into the probability matrix: rows=0..N, cols=y_val_species (true class indices)
    true_class_probs = val_probs[np.arange(len(y_val_species)), y_val_species]
    sample_losses = -np.log(true_class_probs)

    # Calculate correlation between error (loss) and each feature
    correlations = []
    for i, feature_name in enumerate(feature_cols):
        feat_values = X_val[:, i]
        # Handle constant features to avoid warning
        if np.std(feat_values) < 1e-9:
            corr = 0.0
        else:
            corr = np.corrcoef(sample_losses, feat_values)[0, 1]
        correlations.append((feature_name, corr))

    # Sort by magnitude of correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 features correlated with prediction error (Log Loss):")
    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.6f}")

    # 6. Submission Logic
    THRESHOLD = 0.010054905410813797

    if metric < THRESHOLD:
        print("\nMetric meets threshold. Proceeding to full training and submission...")

        # Load full dataset (Train + Val merged) using the library function
        # This function handles the concatenation, scaling, and encoding internally
        print("Loading full dataset...")
        (
            X_full,
            X_test,
            y_species_full,
            y_genus_full,
            test_ids,
            scaler_full,
            species_le_full,
            genus_le_full,
        ) = load_and_process_data(load_cached_data=True)

        # Train on full dataset
        print("Retraining HybridEnsemble on full dataset...")
        final_model = HybridEnsemble()
        final_model.fit(X_full, y_species_full)

        # Predict on Test
        print("Generating test predictions...")
        test_probs = final_model.predict_proba(X_test)
        test_probs = np.clip(test_probs, 1e-15, 1 - 1e-15)

        # Format Submission
        submission_df = pd.DataFrame(test_probs, columns=species_le_full.classes_)
        submission_df.insert(0, ID_COL, test_ids)

        # Save
        submission_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric {metric} is not lower than threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
