import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import warnings
import torch
import random

# Import from provided library
from library.config import RANDOM_SEED, SUBMISSION_DIR, SUBMISSION_PATH, XGB_PARAMS
from library.features import load_train_data, load_val_data, load_test_data
from library.model import StratifiedEnsemble
from library.metrics import calculate_log_mae

# Suppress warnings
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Setup
    set_seed(RANDOM_SEED)
    print("Starting runfile.py execution...")

    # 2. Load Data
    # Using load_cached_data=True as requested to leverage pre-computed features
    print("Loading Training Data...")
    df_train = load_train_data(load_cached_data=True)

    print("Loading Validation Data...")
    df_val = load_val_data(load_cached_data=True)

    # 3. Subsample for Fast Baseline
    # Target around 250,000 samples total for quick training
    TARGET_TRAIN_SIZE = 250000
    if len(df_train) > TARGET_TRAIN_SIZE:
        print(
            f"Subsampling training data from {len(df_train)} to {TARGET_TRAIN_SIZE}..."
        )
        # Stratified sampling to maintain coupling type distribution
        df_train = df_train.groupby("type", group_keys=False).apply(
            lambda x: x.sample(
                n=min(len(x), int(TARGET_TRAIN_SIZE * len(x) / len(df_train))),
                random_state=RANDOM_SEED,
            )
        )

    # 4. Configure Model
    # Adjust params for speed: fewer trees, slightly higher LR
    # XGB_PARAMS in config already includes device='cuda' and tree_method='hist'
    fast_params = XGB_PARAMS.copy()
    fast_params.update(
        {
            "n_estimators": 2000,
            "learning_rate": 0.05,
            "early_stopping_rounds": 50,
            "verbosity": 0,
        }
    )

    model = StratifiedEnsemble(params=fast_params)

    # 5. Train Model
    print("Training Stratified Ensemble...")
    # Prepare inputs
    # Note: StratifiedEnsemble._get_features handles column selection internally
    X_train = df_train
    y_train = df_train["scalar_coupling_constant"]
    groups_train = df_train["type"]

    X_val_full = df_val
    y_val_full = df_val["scalar_coupling_constant"]
    groups_val_full = df_val["type"]

    model.fit(
        X_train,
        y_train,
        groups_train,
        X_val=X_val_full,
        y_val=y_val_full,
        groups_val=groups_val_full,
        verbose=False,
    )

    # 6. Validation Inference
    print("Running Validation Inference...")
    val_preds = model.predict(X_val_full, groups_val_full)

    # 7. Calculate Metric
    # Filter out any potential NaNs (though shouldn't exist for valid types)
    valid_mask = ~val_preds.isna()
    metric = calculate_log_mae(
        y_val_full[valid_mask], val_preds[valid_mask], groups_val_full[valid_mask]
    )

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {metric}")

    # 8. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Create analysis dataframe
    df_analysis = X_val_full[valid_mask].copy()
    df_analysis["abs_error"] = np.abs(y_val_full[valid_mask] - val_preds[valid_mask])

    # Calculate correlation with numeric features
    numeric_cols = df_analysis.select_dtypes(include=[np.number]).columns
    # Exclude IDs and target/error itself
    exclude_analysis = [
        "id",
        "scalar_coupling_constant",
        "abs_error",
        "atom_index_0",
        "atom_index_1",
    ]
    feature_cols = [c for c in numeric_cols if c not in exclude_analysis]

    if feature_cols:
        correlations = (
            df_analysis[feature_cols]
            .corrwith(df_analysis["abs_error"])
            .abs()
            .sort_values(ascending=False)
        )
        print("Top 5 features correlated with error magnitude:")
        print(correlations.head(5))
    else:
        print("No numeric features available for correlation analysis.")

    # 9. Submission Logic
    THRESHOLD = -0.05631934354298804

    if metric < THRESHOLD:
        print(
            f"\nMetric ({metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )

        print("Loading Test Data...")
        df_test = load_test_data(load_cached_data=True)
        groups_test = df_test["type"]

        print("Predicting on Test Set...")
        test_preds = model.predict(df_test, groups_test)

        # Create submission dataframe
        submission = pd.DataFrame(
            {"id": df_test["id"], "scalar_coupling_constant": test_preds}
        )

        # Fill NaNs (if any types were missing in training) with 0.0
        submission["scalar_coupling_constant"] = submission[
            "scalar_coupling_constant"
        ].fillna(0.0)

        # Save
        os.makedirs(SUBMISSION_DIR, exist_ok=True)
        submission.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric ({metric}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
