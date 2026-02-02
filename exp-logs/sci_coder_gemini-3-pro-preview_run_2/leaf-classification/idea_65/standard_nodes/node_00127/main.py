import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

# Import library components
from library.config import (
    RANDOM_SEED,
    SUBMISSION_PATH,
    FLOAT_PRECISION,
    PROB_CLIP_EPS,
    INPUT_DIR,
)
from library.utils import set_seed, clip_log_loss, save_submission
from library.features import get_data
from library.library import get_expert_pool
from library.selection import run_selection


def main():
    # 1. Setup
    set_seed(RANDOM_SEED)

    # 2. Data Loading
    # Load data with morphometric features extracted/cached
    df_train, df_val, df_test = get_data(load_cached_data=True)

    # 3. Label Encoding
    # Ensure we cover all possible classes from train and val for consistent encoding
    le = LabelEncoder()
    all_species = pd.concat([df_train["species"], df_val["species"]]).unique()
    # Sort species to ensure columns are in alphabetical order as per sample submission convention
    le.fit(np.sort(all_species))

    y_train = le.transform(df_train["species"])
    y_val = le.transform(df_val["species"])

    print(f"Classes encoded: {len(le.classes_)}")

    # 4. Phase 1: Expert Training & Selection
    experts = get_expert_pool()
    val_preds_dict = {}

    print(f"Training {len(experts)} experts on training set...")
    for expert in experts:
        # Fit on training data
        expert.fit(df_train, y_train)

        # Predict on validation data
        # Expert.predict_proba returns float64 array
        probs = expert.predict_proba(df_val)
        val_preds_dict[expert.name] = probs

    # Run Greedy Forward Selection
    print("Running Greedy Forward Selection...")
    weights, best_val_loss = run_selection(val_preds_dict, y_val)

    # 5. Validation Reporting
    # REQUIRED: Print the final validation metric in the specific format
    print(f"Final Validation Metric: {best_val_loss}")

    # Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate ensemble predictions on validation set
    ensemble_probs_val = np.zeros(
        (len(df_val), len(le.classes_)), dtype=FLOAT_PRECISION
    )
    total_weight = sum(weights.values())

    for name, w in weights.items():
        ensemble_probs_val += val_preds_dict[name] * w
    ensemble_probs_val /= total_weight

    # Calculate per-sample Cross Entropy Loss for correlation analysis
    # We manually implement the loss calculation to get element-wise values
    # Clip probabilities to avoid log(0)
    probs_clipped = np.clip(ensemble_probs_val, PROB_CLIP_EPS, 1 - PROB_CLIP_EPS)
    # Normalize rows (as per metric definition)
    probs_norm = probs_clipped / probs_clipped.sum(axis=1, keepdims=True)

    # Create one-hot encoded true labels
    y_val_oh = np.zeros_like(probs_norm)
    y_val_oh[np.arange(len(y_val)), y_val] = 1

    # Compute loss per sample
    sample_losses = -np.sum(y_val_oh * np.log(probs_norm), axis=1)

    # Correlate sample loss with numeric features
    numeric_cols = df_val.select_dtypes(include=[np.number]).columns
    correlations = []

    for col in numeric_cols:
        if col == "id":
            continue  # Skip ID

        feat_values = df_val[col].values.astype(float)
        # Handle potential constant columns (std=0) which return nan correlation
        if np.std(feat_values) > 0:
            corr = np.corrcoef(feat_values, sample_losses)[0, 1]
            if not np.isnan(corr):
                correlations.append((col, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error Magnitude:")
    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.4f}")

    # 6. Phase 2: Retraining & Inference
    print("\nRetraining selected experts on full data (Train + Val)...")

    # Combine datasets
    df_full = pd.concat([df_train, df_val]).reset_index(drop=True)
    y_full = np.concatenate([y_train, y_val])

    test_preds_dict = {}

    # Identify selected experts
    unique_selected_experts = set(weights.keys())

    # Re-instantiate experts to ensure clean state and correct config
    # We iterate through the full pool and pick the ones we need
    experts_full_pool = get_expert_pool()
    experts_map = {e.name: e for e in experts_full_pool}

    for name in unique_selected_experts:
        if name in experts_map:
            expert = experts_map[name]
            # Retrain on full data
            expert.fit(df_full, y_full)
            # Predict on test data
            test_probs = expert.predict_proba(df_test)
            test_preds_dict[name] = test_probs
        else:
            print(f"Warning: Selected expert {name} not found in pool.")

    # 7. Weighted Average Inference
    print("Generating final test predictions...")
    ensemble_probs_test = np.zeros(
        (len(df_test), len(le.classes_)), dtype=FLOAT_PRECISION
    )

    for name, w in weights.items():
        if name in test_preds_dict:
            ensemble_probs_test += test_preds_dict[name] * w

    ensemble_probs_test /= total_weight

    # 8. Submission
    threshold = 9.992007221626413e-16

    if best_val_loss < threshold:
        print(
            f"Validation metric ({best_val_loss}) meets the strict threshold ({threshold})."
        )
    else:
        print(
            f"Validation metric ({best_val_loss}) does not meet the strict threshold ({threshold})."
        )
        print("Saving submission regardless to ensure task completion and grading.")

    save_submission(df_test["id"].values, le.classes_, ensemble_probs_test)


if __name__ == "__main__":
    main()
