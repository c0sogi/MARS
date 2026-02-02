import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# Import from provided libraries
from library.config import SEED, TARGET_COL, METADATA_ALLOWLIST, SUBMISSION_PATH, ID_COL
from library.utils import set_seed, Timer
from library.data_loader import load_datasets
from library.features import FeaturePipeline
from library.engine import StackingEngine


def main():
    # 1. Setup
    set_seed(SEED)
    print("Initializing pipeline...")

    # 2. Load Data
    # We load train and val separately to ensure we have a strict hold-out set
    # for the "Final Validation Metric" requirement.
    try:
        train_df, val_df, test_df = load_datasets()
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    print(f"Train shape: {train_df.shape}")
    print(f"Val shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")

    # 3. Feature Engineering
    # Initialize pipeline
    fp = FeaturePipeline()

    # Fit on Train, Transform Train
    # We use a unique cache name to avoid conflicts if re-running
    print("\n--- Feature Engineering: Train ---")
    X_train = fp.fit_transform(train_df, load_cached_data=True, cache_name="run_train")

    # Transform Val
    print("\n--- Feature Engineering: Val ---")
    X_val = fp.transform(val_df, load_cached_data=True, cache_name="run_val")

    # Transform Test
    print("\n--- Feature Engineering: Test ---")
    X_test = fp.transform(test_df, load_cached_data=True, cache_name="run_test")

    # 4. Prepare for Training
    y_train = train_df[TARGET_COL]
    y_val = val_df[TARGET_COL]

    # Generate Folds for StackingEngine
    # StackingEngine expects a list of (train_idx, val_idx) tuples
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.zeros(len(y_train)), y_train))

    # 5. Train Stacking Engine
    engine = StackingEngine()

    print("\n--- Starting Engine Training ---")
    # Note: We are training on the 'train_df' partition.
    # The engine will perform internal CV (5-fold) on this data to train base models and the meta-learner.
    engine.train(X_train, y_train, folds)

    # 6. Validation Inference
    print("\n--- Performing Validation Inference ---")
    # The engine uses hybrid inference: averaging volatile fold models and using retrained stable models
    val_preds = engine.predict(X_val)

    # Calculate Metric
    final_auc = roc_auc_score(y_val, val_preds)
    print(f"Final Validation Metric: {final_auc}")

    # 7. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate error magnitude
    errors = np.abs(y_val - val_preds)

    # Correlate with metadata features (dense features are easier to interpret)
    # X_val['meta'] corresponds to METADATA_ALLOWLIST (scaled)
    meta_features = X_val["meta"]

    print("Correlation between Error Magnitude and Metadata Features:")
    feature_correlations = []
    for i, feature_name in enumerate(METADATA_ALLOWLIST):
        if i < meta_features.shape[1]:
            feat_values = meta_features[:, i]
            # Handle constant features to avoid warnings
            if np.std(feat_values) == 0:
                corr = 0.0
            else:
                corr, _ = pearsonr(errors, feat_values)
            feature_correlations.append((feature_name, corr))
            print(f"  {feature_name}: {corr:.4f}")

    # Sort by absolute correlation
    feature_correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    print(
        f"\nTop factor associated with error: {feature_correlations[0][0]} ({feature_correlations[0][1]:.4f})"
    )

    # 8. Submission
    THRESHOLD = 0.7222984867326668

    if final_auc > THRESHOLD:
        print(
            f"\nValidation metric ({final_auc}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        test_preds = engine.predict(X_test)

        submission_df = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET_COL: test_preds})

        # Ensure directory exists
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

        submission_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
        print(f"Submission shape: {submission_df.shape}")
        print("Head of submission:")
        print(submission_df.head())
    else:
        print(
            f"\nValidation metric ({final_auc}) does NOT meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
