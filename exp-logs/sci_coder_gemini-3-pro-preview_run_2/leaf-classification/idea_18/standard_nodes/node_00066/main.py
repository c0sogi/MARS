import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

# Add library path to system path to import provided modules
sys.path.append("./library")

import data_interface
import expert_pipelines
import ensemble_optimizer


def run():
    # ==================================================================================
    # 1. Setup and Data Loading
    # ==================================================================================
    print("Initializing ADGE workflow...")

    # Load stratified split (Train/Val)
    # Using cached data for speed as per instructions
    X_train, y_train, X_val, y_val = data_interface.get_stratified_split(
        load_cached_data=True
    )

    # Load test data for final submission
    _, _, _, _, X_test, test_ids, classes = data_interface.load_dataset(
        load_cached_data=True
    )

    print(
        f"Data Loaded: Train shape {X_train.shape}, Val shape {X_val.shape}, Test shape {X_test.shape}"
    )

    # ==================================================================================
    # 2. Phase 1: Expert Training & Selection
    # ==================================================================================
    print("\n--- Phase 1: Expert Training & Selection ---")

    # Define the pool of experts
    experts = {
        "global_lda": expert_pipelines.build_global_lda(),
        "denoised_lda_144": expert_pipelines.build_denoised_lda(k_features=144),
        "denoised_lda_96": expert_pipelines.build_denoised_lda(k_features=96),
        "denoised_lda_48": expert_pipelines.build_denoised_lda(k_features=48),
        "global_lr": expert_pipelines.build_global_lr(),
    }

    val_preds_dict = {}
    trained_experts_phase1 = {}

    # Train each expert and predict on validation set
    for name, model in experts.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        # Predict on validation set
        preds = model.predict_proba(X_val)
        val_preds_dict[name] = preds
        trained_experts_phase1[name] = model

        # Individual performance check (optional, for logging)
        loss = log_loss(y_val, preds, labels=np.arange(len(classes)))
        print(f"  -> {name} Val Log Loss: {loss:.6f}")

    # Run Greedy Ensemble Selection
    print("\nRunning Greedy Ensemble Selection...")
    # Max iterations 20 is sufficient to find a stable ensemble
    selector = ensemble_optimizer.GreedyEnsembleSelector(
        max_iterations=20, tolerance=1e-6
    )
    selector.fit(val_preds_dict, y_val)

    # Get optimal weights and selected models
    weights = selector.weights_
    selected_model_names = list(weights.keys())

    # Generate Ensemble Validation Predictions
    val_ensemble_preds = selector.predict(val_preds_dict)

    # ==================================================================================
    # 3. Validation Metric
    # ==================================================================================
    # Calculate final metric on the hold-out validation set
    final_val_metric = log_loss(
        y_val, val_ensemble_preds, labels=np.arange(len(classes))
    )
    print(f"Final Validation Metric: {final_val_metric}")

    # ==================================================================================
    # 4. Failure Analysis
    # ==================================================================================
    print("\n--- Failure Analysis ---")

    # Calculate per-sample log loss
    # Clip predictions to avoid log(0)
    epsilon = 1e-15
    clipped_preds = np.clip(val_ensemble_preds, epsilon, 1 - epsilon)

    # Get the probability assigned to the true class for each sample
    # y_val contains integer class indices
    true_class_probs = clipped_preds[np.arange(len(y_val)), y_val]

    # Compute negative log likelihood (loss) for each sample
    sample_losses = -np.log(true_class_probs)

    # Correlate sample loss with each feature in X_val
    correlations = []
    n_features = X_val.shape[1]

    for i in range(n_features):
        feat_values = X_val[:, i]
        # Skip constant features to avoid warnings
        if np.std(feat_values) == 0:
            corr = 0
        else:
            corr, _ = pearsonr(feat_values, sample_losses)
        correlations.append((i, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error (Log Loss):")
    for idx, corr in correlations[:5]:
        print(f"  Feature {idx}: Correlation = {corr:.4f}")

    # ==================================================================================
    # 5. Phase 2: Final Retraining & Submission
    # ==================================================================================
    # Threshold defined in task
    threshold = 3.3768e-06

    if final_val_metric < threshold:
        print(
            f"\nMetric ({final_val_metric}) meets threshold ({threshold}). Proceeding to submission."
        )

        # Combine Train and Validation sets for maximum data utilization
        X_full = np.vstack((X_train, X_val))
        y_full = np.concatenate((y_train, y_val))

        test_preds_dict = {}

        # Retrain only the selected experts
        for name in selected_model_names:
            print(f"Retraining selected expert: {name} on full data...")

            if name == "global_lr":
                # Extract best C from Phase 1 model to prevent drift
                # The pipeline structure is 'scaler' -> 'lr_cv'
                lr_cv_step = trained_experts_phase1[name].named_steps["lr_cv"]

                # C_ is an array. For multinomial lbfgs, it typically contains the single best C.
                # We take the first element.
                best_c = lr_cv_step.C_[0]
                print(f"  -> Extracted Best C for LR: {best_c}")

                # Build fixed LR pipeline
                model = expert_pipelines.build_fixed_lr(C=best_c)

            elif "denoised_lda" in name:
                # Rebuild denoised LDA with the specific k
                if "144" in name:
                    k = 144
                elif "96" in name:
                    k = 96
                elif "48" in name:
                    k = 48
                else:
                    k = 192  # Fallback, though shouldn't happen
                model = expert_pipelines.build_denoised_lda(k_features=k)

            elif name == "global_lda":
                model = expert_pipelines.build_global_lda()

            else:
                raise ValueError(f"Unknown model name: {name}")

            # Fit on full dataset
            model.fit(X_full, y_full)

            # Generate predictions for Test set
            test_preds_dict[name] = model.predict_proba(X_test)

        # Ensemble Test Predictions
        # The selector uses the weights learned in Phase 1
        final_test_preds = selector.predict(test_preds_dict)

        # Save Submission
        print("Saving submission...")
        output_path = "./submission/submission.csv"
        data_interface.save_submission(
            final_test_preds, test_ids, classes, output_path=output_path
        )
        print(f"Submission saved successfully to {output_path}")

    else:
        print(
            f"\nMetric ({final_val_metric}) does NOT meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    run()
