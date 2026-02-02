import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Add current directory to sys.path to ensure library modules are found
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import (
    set_seed,
    save_submission,
    log_loss_score,
    normalize_probabilities,
    clip_probabilities,
)
from library.data_loader import load_dataset, get_class_names
from library.expert_library import get_expert_pool
from library.ensemble import GreedySelector


def main():
    # 1. Initialization
    Config.initialize()
    set_seed(Config.RANDOM_SEED)

    print("Initializing DMGE Workflow...")

    # 2. Data Loading
    # Load separate splits
    # We use cached data if available to speed up execution
    train_data = load_dataset("train", load_cached_data=True)
    val_data = load_dataset("val", load_cached_data=True)
    test_data = load_dataset("test", load_cached_data=True)
    class_names = get_class_names(load_cached_data=True)

    # Prepare Combined Data for Final Refit (Phase 2)
    # Concatenate Global Features
    X_global_full = np.concatenate(
        [train_data["X_global"], val_data["X_global"]], axis=0
    )
    # Concatenate Morphometric Features
    X_morph_full = np.concatenate([train_data["X_morph"], val_data["X_morph"]], axis=0)
    # Concatenate Labels
    y_full = np.concatenate([train_data["y"], val_data["y"]], axis=0)

    full_data = {"X_global": X_global_full, "X_morph": X_morph_full}

    # 3. Expert Library & Ensemble Setup
    experts = get_expert_pool()
    selector = GreedySelector(experts)

    # 4. Phase 1: Selection (Train on Train, Evaluate on Val)
    print("\n--- Phase 1: Greedy Forward Selection ---")
    # selector.fit() trains candidates on train_data and selects based on val_data performance
    selector.fit(train_data, train_data["y"], val_data, val_data["y"])

    # 5. Validation Assessment & Failure Analysis
    print("\n--- Validation Assessment ---")

    if not selector.selected_experts:
        print("No experts selected. Aborting.")
        return

    # Encode labels for numerical metric calculation
    le = LabelEncoder()
    le.fit(train_data["y"])
    y_val_indices = le.transform(val_data["y"])

    n_val = len(val_data["y"])
    n_classes = len(class_names)
    val_probs_sum = np.zeros((n_val, n_classes), dtype=Config.FLOAT_TYPE)

    print(
        f"Re-evaluating {len(selector.selected_experts)} selected experts on validation set for analysis..."
    )

    # Manually retrain selected experts on TRAIN split to reproduce validation predictions
    for expert in selector.selected_experts:
        # Build pipeline
        pipeline = expert.build_pipeline()
        # Fit on Train
        X_train_view = train_data[expert.view_name]
        pipeline.fit(X_train_view, train_data["y"])
        # Predict on Val
        X_val_view = val_data[expert.view_name]
        preds = pipeline.predict_proba(X_val_view).astype(Config.FLOAT_TYPE)
        val_probs_sum += preds

    # Average probabilities (Ensemble prediction)
    val_probs = val_probs_sum / len(selector.selected_experts)

    # Compute Final Metric
    final_metric = log_loss_score(y_val_indices, val_probs)
    # REQUIRED PRINT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    print("\n--- Failure Analysis ---")
    # 1. Calculate per-sample Log Loss
    # Normalize and clip exactly as in the metric function to ensure consistency
    val_probs_norm = normalize_probabilities(val_probs)
    val_probs_clipped = clip_probabilities(val_probs_norm)

    # Extract probability assigned to the true class
    # rows: 0..N-1, cols: y_val_indices
    true_class_probs = val_probs_clipped[np.arange(n_val), y_val_indices]
    sample_losses = -np.log(true_class_probs)

    # 2. Correlation with Global Features
    # We correlate the error vector (sample_losses) with each feature in X_global (val)
    X_val_global = val_data["X_global"]
    n_features = X_val_global.shape[1]
    correlations = []

    for i in range(n_features):
        feature_vec = X_val_global[:, i]
        # Handle potential constant features (std=0) to avoid nan
        if np.std(feature_vec) < 1e-9:
            corr = 0.0
        else:
            corr = np.corrcoef(sample_losses, feature_vec)[0, 1]

        # We record the correlation
        correlations.append(corr)

    correlations = np.array(correlations)

    # Get top 5 features most associated with error (highest absolute correlation)
    top_indices = np.argsort(np.abs(correlations))[::-1][:5]

    print("Top 5 Features correlated with Model Error (Log Loss):")
    for idx in top_indices:
        print(f"  Feature Index {idx}: Correlation = {correlations[idx]:.4f}")

    # 6. Phase 2: Final Refit (Train on Full Data)
    print("\n--- Phase 2: Final Refit ---")
    # Retrain the selected experts on the combined Train + Val dataset
    selector.refit(full_data, y_full)

    # 7. Submission Generation
    # We use a safe upper bound (10.0) to ensure a submission is generated
    # unless the model is completely divergent.
    THRESHOLD = 10.0

    if final_metric < THRESHOLD:
        print("\nGenerating predictions for test set...")
        test_preds = selector.predict(test_data)

        print("Saving submission...")
        save_submission(test_data["ids"], class_names, test_preds)
    else:
        print(
            f"Validation metric {final_metric} is too high (Threshold: {THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
