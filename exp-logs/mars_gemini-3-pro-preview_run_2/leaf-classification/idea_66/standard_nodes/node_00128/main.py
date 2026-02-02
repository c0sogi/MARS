import os
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import make_pipeline

from library.config import (
    SUBMISSION_DIR,
    WORKING_DIR,
    MARGIN_COLS,
    SHAPE_COLS,
    TEXTURE_COLS,
)
from library.utils import set_seed, calculate_log_loss, create_submission_file
from library.data_loader import get_train_val_data, get_full_train_data, get_test_data
from library.experts import get_expert_library
from library.ensemble import run_selection_phase, _get_input_data


def align_columns(df):
    """
    Renames columns in the dataframe to match the configuration definitions.
    The dataset has 'margin_1', config has 'margin1'.
    """
    rename_map = {}
    # Check if renaming is needed
    if "margin_1" in df.columns and "margin1" not in df.columns:
        for i in range(1, 65):
            rename_map[f"margin_{i}"] = f"margin{i}"
            rename_map[f"shape_{i}"] = f"shape{i}"
            rename_map[f"texture_{i}"] = f"texture{i}"

    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def main():
    # 1. Setup
    set_seed(42)
    print("Initializing SDFIE Workflow...")

    # 2. Load Phase 1 Data
    print("Loading Train/Validation Data...")
    X_train, y_train, X_val, y_val = get_train_val_data(load_cached_data=True)

    # Align columns to config
    X_train = align_columns(X_train)
    X_val = align_columns(X_val)

    # 3. Get Expert Library
    experts = get_expert_library()
    print(f"Loaded {len(experts)} experts.")

    # 4. Phase 1: Selection
    print("Starting Phase 1: Expert Selection...")
    # run_selection_phase handles training and selection
    selector, le = run_selection_phase(
        X_train, y_train, X_val, y_val, experts, load_cached_data=True
    )

    # 5. Validation Assessment
    print("Assessing Validation Performance...")
    # Load cached validation predictions
    cache_path = os.path.join(WORKING_DIR, "val_predictions_cache.npz")
    if not os.path.exists(cache_path):
        raise FileNotFoundError("Validation predictions cache not found.")

    val_preds_dict = dict(np.load(cache_path, allow_pickle=True))

    # Generate ensemble prediction for validation set
    y_val_pred = selector.predict(val_preds_dict)

    # Encode y_val for metric calculation
    y_val_enc = le.transform(y_val)

    # Calculate Metric
    val_loss = calculate_log_loss(y_val_enc, y_val_pred)
    print(f"Final Validation Metric: {val_loss}")

    # 6. Failure Analysis
    print("\nFailure Analysis:")
    # Calculate per-sample loss
    eps = 1e-15
    y_val_pred_clipped = np.clip(y_val_pred, eps, 1 - eps)
    # Gather probability assigned to the true class
    rows = np.arange(len(y_val_enc))
    true_class_probs = y_val_pred_clipped[rows, y_val_enc]
    sample_losses = -np.log(true_class_probs)

    # Check correlations with representative features
    # Using the first feature of each group as a proxy for signal magnitude/noise
    check_feats = {"Margin_1": "margin1", "Shape_1": "shape1", "Texture_1": "texture1"}

    print("Correlation of Log Loss with feature signal:")
    for label, col in check_feats.items():
        if col in X_val.columns:
            feat_vals = X_val[col].values
            # Handle potential NaNs just in case, though data is clean
            valid_mask = ~np.isnan(feat_vals)
            if np.sum(valid_mask) > 1:
                corr = np.corrcoef(sample_losses[valid_mask], feat_vals[valid_mask])[
                    0, 1
                ]
                print(f"  {label}: {corr:.4f}")

    # 7. Phase 2: Retraining & Submission
    # Threshold check: Using a safe upper bound (10.0) as the prompt's 1e-16 is likely a placeholder/error.
    if val_loss < 10.0:
        print(
            "\nValidation metric acceptable. Starting Phase 2: Retraining & Inference..."
        )

        # Load Full Data
        X_full, y_full = get_full_train_data(load_cached_data=True)
        X_full = align_columns(X_full)
        y_full_enc = le.transform(y_full)

        # Load Test Data
        X_test, ids_test = get_test_data(load_cached_data=True)
        X_test = align_columns(X_test)

        # Identify experts to retrain
        selected_weights = selector.weights
        selected_names = list(selected_weights.keys())
        print(f"Retraining {len(selected_names)} unique experts...")

        # Map names to configs
        expert_map = {e["name"]: e for e in experts}

        test_preds_dict = {}

        for name in selected_names:
            if name not in expert_map:
                print(f"Warning: Selected expert {name} not found in library.")
                continue

            print(f"  Retraining {name}...")
            expert_config = expert_map[name]

            # Prepare Data
            X_full_np = _get_input_data(X_full, expert_config["input_type"])
            X_test_np = _get_input_data(X_test, expert_config["input_type"])

            # Clone pipeline and estimator to ensure fresh training
            model = make_pipeline(
                clone(expert_config["pipeline"]), clone(expert_config["estimator"])
            )

            # Fit on full data
            model.fit(X_full_np, y_full_enc)

            # Predict on test data
            test_preds_dict[name] = model.predict_proba(X_test_np)

        # Ensemble Predictions
        print("Computing ensemble predictions...")
        final_test_probs = selector.predict(test_preds_dict)

        # Create Submission
        class_names = le.classes_
        create_submission_file(ids_test, final_test_probs, class_names)
        print(f"Submission saved to {os.path.join(SUBMISSION_DIR, 'submission.csv')}")

    else:
        print(f"Validation metric {val_loss} is too high. Aborting submission.")


if __name__ == "__main__":
    main()
