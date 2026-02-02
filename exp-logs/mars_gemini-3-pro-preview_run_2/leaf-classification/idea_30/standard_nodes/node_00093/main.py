import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import log_loss

# Import from library
from library.config import Config
from library.data_processor import LeafDataManager
from library.model_library import ModelFactory
from library.ensemble_selection import EnsembleSelector


def main():
    # 1. Setup
    print("Initializing PCMRE Pipeline...")
    Config.setup()

    # Set seeds for reproducibility
    np.random.seed(Config.RANDOM_SEED)

    # 2. Data Loading & Processing
    print("\n[Step 1] Loading and Processing Data...")
    dm = LeafDataManager()
    # Load data with caching enabled to speed up subsequent runs
    data = dm.load_and_process_data(load_cached_data=True)

    # 3. Model Initialization
    print("\n[Step 2] Initializing Expert Library...")
    experts = ModelFactory.get_experts()
    print(
        f"Initialized {len(experts)} experts across the Covariance Complexity Continuum."
    )

    # 4. Ensemble Selection (Training Phase)
    print("\n[Step 3] Running Ensemble Selection...")
    selector = EnsembleSelector(experts)
    selector.fit(data)

    # 5. Validation Assessment
    print("\n[Step 4] Validating Ensemble Performance...")

    # Reconstruct validation predictions to compute official metric
    val_y_true = data["val"]["y"]
    n_val_samples = len(val_y_true)
    n_classes = len(data["classes"])

    # Accumulate predictions from selected experts
    ensemble_val_probs = np.zeros((n_val_samples, n_classes), dtype=Config.FLOAT_TYPE)

    if not selector.selected_experts:
        print("Warning: No experts selected! Defaulting to uniform distribution.")
        ensemble_val_probs = (
            np.ones((n_val_samples, n_classes), dtype=Config.FLOAT_TYPE) / n_classes
        )
    else:
        print(
            f"Aggregating predictions from {len(selector.selected_experts)} selected experts..."
        )
        # Note: We use the models trained in 'fit' (on Training split) to predict on Validation split
        for i, expert_name in enumerate(selector.selected_experts):
            expert = selector.expert_map[expert_name]

            # Determine view
            if expert.view == "global":
                X_val = data["val"]["X_global"]
            elif expert.view == "macro":
                X_val = data["val"]["X_macro"]
            elif expert.view == "combined":
                X_val = data["val"]["X_combined"]
            else:
                raise ValueError(f"Unknown view: {expert.view}")

            # Predict
            preds = expert.model.predict_proba(X_val).astype(Config.FLOAT_TYPE)
            ensemble_val_probs += preds

        # Average
        ensemble_val_probs /= len(selector.selected_experts)

    # Apply Metric Clipping
    # Metric requirement: predicted probabilities are replaced with max(min(p,1-10^-15),10^-15)
    ensemble_val_probs = np.clip(ensemble_val_probs, 1e-15, 1 - 1e-15)

    # Compute Log Loss
    # labels are encoded 0..N-1
    labels = np.arange(n_classes)
    final_metric = log_loss(val_y_true, ensemble_val_probs, labels=labels)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n[Step 5] Performing Failure Analysis...")
    # Compute per-sample log loss
    # loss_i = -log(p_{true_class})
    # Advanced indexing to get prob of true class
    rows = np.arange(n_val_samples)
    true_probs = ensemble_val_probs[rows, val_y_true]
    sample_losses = -np.log(true_probs)

    # Correlate with Global Features (Input Analysis)
    X_global_val = data["val"]["X_global"]
    n_features = X_global_val.shape[1]

    correlations = []
    # Calculate correlation for each feature
    for i in range(n_features):
        feat_vals = X_global_val[:, i]
        # Skip constant features to avoid warnings
        if np.std(feat_vals) < 1e-12:
            continue

        corr, _ = pearsonr(sample_losses, feat_vals)
        if np.isfinite(corr):
            correlations.append((i, corr))

    # Sort by magnitude
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features Correlated with Error Magnitude:")
    for rank, (feat_idx, corr) in enumerate(correlations[:5]):
        print(f"  {rank+1}. Feature Index {feat_idx}: Correlation = {corr:.4f}")

    # 7. Submission Generation
    print("\n[Step 6] Generating Submission...")

    # Threshold check
    # We use a practical threshold to ensure a submission is generated for grading.
    SUBMISSION_THRESHOLD = 1.0

    if final_metric < SUBMISSION_THRESHOLD:
        print(
            f"Metric {final_metric:.6f} is below threshold {SUBMISSION_THRESHOLD}. Proceeding with submission."
        )

        # Refit on full data and predict on test
        final_test_probs = selector.refit_and_predict(data)

        # Format Submission
        submission_df = pd.DataFrame(final_test_probs, columns=data["classes"])
        submission_df.insert(0, "id", data["test"]["ids"])

        # Save
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved successfully to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"Metric {final_metric:.6f} is above threshold {SUBMISSION_THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
