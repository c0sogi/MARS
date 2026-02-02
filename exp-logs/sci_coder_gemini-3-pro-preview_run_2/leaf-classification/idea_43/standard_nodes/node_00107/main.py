import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from scipy.stats import pearsonr
from collections import Counter

# Import provided library modules
from library.config import RANDOM_SEED, WORKING_DIR, SUBMISSION_FILE, FLOAT_PRECISION
from library.data import get_datasets
from library.ensemble import (
    run_phase_1_selection,
    run_phase_2_inference,
    postprocess_probabilities,
)

# Set fixed random seed
np.random.seed(RANDOM_SEED)


def main():
    print("Starting orchestration script...")

    # --------------------------------------------------------------------------
    # 1. Load Validation Data
    # --------------------------------------------------------------------------
    # We load the 'global' view to get the validation labels (y_val) and
    # features (X_val_global) for failure analysis.
    print("Loading validation data...")
    _, (X_val_global, y_val, val_ids), _, _, classes = get_datasets(
        view="global", load_cached_data=True
    )

    # --------------------------------------------------------------------------
    # 2. Phase 1: Library Generation & Ensemble Selection
    # --------------------------------------------------------------------------
    print("Running Phase 1: Expert Selection...")
    # This function trains experts on Train, predicts on Val, and selects the best subset.
    selected_keys = run_phase_1_selection(load_cached_data=True)

    if not selected_keys:
        print("Error: No experts were selected. Aborting.")
        return

    # --------------------------------------------------------------------------
    # 3. Validation Assessment
    # --------------------------------------------------------------------------
    print("Reconstructing ensemble predictions for validation assessment...")

    # Initialize accumulator for weighted probabilities
    val_preds_accum = np.zeros((len(y_val), len(classes)), dtype=FLOAT_PRECISION)

    # Count occurrences of each expert (Greedy selection allows duplicates which act as weights)
    expert_counts = Counter(selected_keys)
    total_weight = sum(expert_counts.values())

    # Load cached predictions for selected experts and aggregate
    for key, count in expert_counts.items():
        pred_path = os.path.join(WORKING_DIR, f"pred_val_{key}.npy")
        if os.path.exists(pred_path):
            # Load and weight
            preds = np.load(pred_path).astype(FLOAT_PRECISION)
            val_preds_accum += preds * count
        else:
            print(f"Warning: Cached prediction for expert '{key}' not found.")
            # In a strict pipeline, we might raise an error, but here we continue
            total_weight -= count

    if total_weight == 0:
        print("Error: Total ensemble weight is zero.")
        return

    # Normalize by total weight to get average
    val_preds_ensemble = val_preds_accum / total_weight

    # Apply competition-specific post-processing (Normalize & Clip)
    val_preds_ensemble = postprocess_probabilities(val_preds_ensemble)

    # Calculate Final Metric
    final_metric = log_loss(y_val, val_preds_ensemble)

    # Print required metric string
    print(f"Final Validation Metric: {final_metric}")

    # --------------------------------------------------------------------------
    # 4. Failure Analysis
    # --------------------------------------------------------------------------
    print("Performing Failure Analysis...")

    # Calculate Log Loss per sample: -log(p_true_class)
    # Get probability assigned to the true class
    row_indices = np.arange(len(y_val))
    true_class_probs = val_preds_ensemble[row_indices, y_val]

    # Clip for safety (though postprocess_probabilities already did this)
    epsilon = 1e-15
    true_class_probs = np.clip(true_class_probs, epsilon, 1 - epsilon)

    sample_losses = -np.log(true_class_probs)

    # Correlate sample error with global features
    # This helps identify if specific shapes/textures/margins are associated with failure
    feature_correlations = []
    n_features = X_val_global.shape[1]

    for i in range(n_features):
        feat_vals = X_val_global[:, i]
        # Skip constant features to avoid warnings
        if np.std(feat_vals) < 1e-12:
            corr = 0.0
        else:
            corr, _ = pearsonr(sample_losses, feat_vals)
        feature_correlations.append((i, corr))

    # Sort by magnitude of correlation (descending)
    feature_correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error Magnitude:")
    for i, (feat_idx, corr) in enumerate(feature_correlations[:5]):
        print(f"  Feature {feat_idx}: Correlation = {corr:.6f}")

    # --------------------------------------------------------------------------
    # 5. Phase 2: Inference & Submission
    # --------------------------------------------------------------------------
    # Check threshold condition mentioned in task
    threshold = 9.992007221626413e-16
    if final_metric < threshold:
        print(
            f"Validation metric meets the strict threshold ({final_metric} < {threshold})."
        )
    else:
        print(
            f"Validation metric ({final_metric}) is above strict threshold ({threshold}). Proceeding with submission generation to satisfy output requirements."
        )

    print("Running Phase 2: Retraining and Inference...")

    # Retrain selected experts on Full Train and predict on Test
    test_ids, test_preds, class_names = run_phase_2_inference(
        selected_keys, load_cached_data=True
    )

    # Construct Submission DataFrame
    df_sub = pd.DataFrame(test_preds, columns=class_names)
    df_sub.insert(0, "id", test_ids)

    # Save to CSV
    print(f"Saving submission file to {SUBMISSION_FILE}...")
    df_sub.to_csv(SUBMISSION_FILE, index=False)
    print("Submission saved successfully.")
    print("Run complete.")


if __name__ == "__main__":
    main()
