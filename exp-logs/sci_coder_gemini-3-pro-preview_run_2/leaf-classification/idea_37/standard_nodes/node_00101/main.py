import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Ensure library is in path
sys.path.append(os.getcwd())

from library.config import Config
from library.data_loader import DataLoader
from library.model_factory import ModelFactory
from library.ensemble_selection import GreedyEnsembleSelector
from library.utils import calculate_log_loss, save_submission, load_metadata


def run():
    # 1. Setup
    print("Initializing Constrained-Basis Dual-Stream Generative Ensemble...")
    Config.setup()

    # 2. Load Phase 1 Data (Train/Val Split)
    print("\n--- Phase 1: Data Loading ---")
    loader = DataLoader()
    train_data, val_data, classes = loader.load_phase1_data(load_cached_data=True)

    # 3. Model Selection
    print("\n--- Phase 1: Ensemble Selection ---")
    experts = ModelFactory.generate_expert_library()
    # Using 50 iterations as a robust upper bound for greedy selection
    selector = GreedyEnsembleSelector(max_iterations=50, tolerance=1e-6)
    selector.fit(train_data, val_data, experts, load_cached_data=True)

    selected_weights = selector.get_selected_config()
    best_score = selector.best_score_

    # REQUIRED OUTPUT: Final Validation Metric
    # Printed with full precision
    print(f"Final Validation Metric: {best_score}")

    # 4. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Reconstruct validation predictions to compute per-sample loss
    # Load cached predictions from selector to avoid re-inference
    preds_path = os.path.join(Config.WORKING_DIR, "p1_library_val_preds.npy")
    ids_path = os.path.join(Config.WORKING_DIR, "p1_library_ids.npy")

    if os.path.exists(preds_path) and os.path.exists(ids_path):
        all_preds = np.load(preds_path)
        expert_ids = np.load(ids_path, allow_pickle=True)

        # Map ID to index in the cached array
        id_to_idx = {eid: i for i, eid in enumerate(expert_ids)}

        # Construct ensemble prediction
        n_samples = val_data["y"].shape[0]
        n_classes = len(classes)
        ensemble_preds = np.zeros((n_samples, n_classes), dtype=Config.NP_DTYPE)
        total_weight = 0

        for eid, weight in selected_weights.items():
            if eid in id_to_idx:
                idx = id_to_idx[eid]
                ensemble_preds += all_preds[idx] * weight
                total_weight += weight

        if total_weight > 0:
            ensemble_preds /= total_weight

            # Calculate per-sample log loss
            y_val = val_data["y"]
            # Clip for numerical stability (consistent with metric definition)
            eps = 1e-15
            ensemble_preds_clipped = np.clip(ensemble_preds, eps, 1 - eps)

            # Gather probability assigned to the true class
            # y_val are indices
            prob_true = ensemble_preds_clipped[np.arange(n_samples), y_val]
            sample_losses = -np.log(prob_true)

            # Load raw validation metadata for feature correlation
            df_val = load_metadata("val")
            # Exclude non-feature columns
            feature_cols = [
                c for c in df_val.columns if c not in ["id", "species", "image_path"]
            ]

            correlations = []
            for col in feature_cols:
                feat_values = df_val[col].values
                # Check for constant features to avoid warnings
                if np.std(feat_values) > 1e-9:
                    corr, _ = pearsonr(sample_losses, feat_values)
                    correlations.append((col, corr))

            # Sort by absolute correlation
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)

            print("Top 5 Features correlated with Error (Log Loss):")
            for name, corr in correlations[:5]:
                print(f"  {name}: {corr:.4f}")
        else:
            print("Warning: No experts selected or weights are zero.")
    else:
        print("Could not load cached predictions for failure analysis.")

    # 5. Phase 2: Final Retraining & Submission
    # We use a practical threshold of 2.0 to ensure submission generation for valid models.
    THRESHOLD = 2.0

    if best_score < THRESHOLD:
        print("\n--- Phase 2: Final Retraining & Submission ---")

        # Load Phase 2 Data (Full Train + Test)
        full_data, test_data, classes_p2 = loader.load_phase2_data(
            load_cached_data=True
        )

        # Initialize final ensemble accumulator
        n_test = len(test_data["ids"])
        n_classes_final = len(classes_p2)
        final_sum_probs = np.zeros((n_test, n_classes_final), dtype=Config.NP_DTYPE)
        total_weight = 0

        # Instantiate fresh experts for retraining
        all_experts = ModelFactory.generate_expert_library()

        print(
            f"Retraining {len(selected_weights)} selected expert configuration(s) on full data..."
        )

        for expert_def in all_experts:
            eid = expert_def["id"]
            if eid in selected_weights:
                weight = selected_weights[eid]
                model = expert_def["model"]
                stream_name = expert_def["stream"]

                # Train on Full Data (Train + Val)
                X_full = full_data[stream_name]
                y_full = full_data["y"]
                model.fit(X_full, y_full)

                # Predict on Test Data
                X_test = test_data[stream_name]
                probs = model.predict_proba(X_test).astype(Config.NP_DTYPE)

                # Accumulate
                final_sum_probs += probs * weight
                total_weight += weight

        if total_weight > 0:
            final_probs = final_sum_probs / total_weight

            # Save Submission
            save_submission(test_data["ids"], classes_p2, final_probs)
            print(f"Submission saved to {Config.SUBMISSION_FILE_PATH}")
        else:
            print("Error: Total weight is zero.")

    else:
        print(
            f"Validation score {best_score} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    run()
