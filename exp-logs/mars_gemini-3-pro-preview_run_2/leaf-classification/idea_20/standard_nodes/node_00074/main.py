import sys
import os
import numpy as np
import pandas as pd

# Add library path to ensure imports work correctly
sys.path.append("./library")

import library.data_loader as dl
import library.models as models
import library.ensemble as ensemble
import library.utils as utils


def main():
    # 1. Setup and Reproducibility
    utils.set_seed(42)

    # 2. Data Loading
    # Uses the provided data_loader which applies Global Gaussianization (PowerTransformer)
    print("Loading and preprocessing data...")
    # Force reload (load_cached_data=False) to ensure fresh float64 processing
    # Cite solution_lesson_node_00073
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = dl.load_data(
        load_cached_data=False
    )

    # Create numeric class indices (0 to 98) for metric calculation
    class_indices = np.arange(len(classes))

    # 3. Phase 1: Expert Training & Selection
    print("Phase 1: Training Experts and running Selection...")

    # Only using Expert_A (LDA) as it is optimal for this task
    # Cite solution_lesson_node_00072, solution_lesson_node_00056
    expert_names = ["Expert_A"]
    trained_models = {}
    val_preds = {}

    # Train each expert on the training split
    for name in expert_names:
        print(f"Training {name}...")
        # Get the pipeline (includes PowerTransformer -> Model)
        # Note: Input data is already transformed by load_data, and pipeline transforms it again.
        # We strictly follow the provided architecture files.
        model = models.get_expert_pipeline(name, random_state=42)
        model.fit(X_train, y_train)

        # Generate validation probabilities
        preds = model.predict_proba(X_val)
        val_preds[name] = preds
        trained_models[name] = model

    # Select optimal ensemble using Greedy Forward Selection
    print("Running Greedy Forward Selection on Validation Set...")
    selector = ensemble.GreedySelector(tolerance=1e-6)
    selector.fit(val_preds, y_val, classes=class_indices)

    selected_experts = selector.selected_experts
    best_val_score = selector.best_score

    # Required Output: Final Validation Metric
    print(f"Final Validation Metric: {best_val_score}")

    # 4. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Get ensemble predictions on validation set (simple average of selected experts)
    ensemble_val_probs = selector.predict(val_preds)

    # Clip and Normalize to match metric calculation logic
    ensemble_val_probs = np.clip(ensemble_val_probs, 1e-15, 1 - 1e-15)
    row_sums = ensemble_val_probs.sum(axis=1, keepdims=True)
    ensemble_val_probs = ensemble_val_probs / row_sums

    # Calculate per-sample Log Loss
    # y_val contains the true class indices
    true_class_probs = ensemble_val_probs[np.arange(len(y_val)), y_val]
    sample_losses = -np.log(true_class_probs)

    # Correlate error with features
    correlations = []
    n_features = X_val.shape[1]

    for i in range(n_features):
        feat_values = X_val[:, i]
        # Avoid division by zero in correlation if feature is constant
        if np.std(feat_values) > 0:
            corr = np.corrcoef(feat_values, sample_losses)[0, 1]
            correlations.append((i, corr))
        else:
            correlations.append((i, 0.0))

    # Sort by magnitude of correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Prediction Error:")
    for idx, corr in correlations[:5]:
        print(f"  Feature Index {idx}: Correlation = {corr:.4f}")

    # 5. Phase 2: Final Retraining & Submission
    THRESHOLD = 1.4705545687989736e-08

    if best_val_score < THRESHOLD:
        print(
            f"\nValidation score meets threshold ({THRESHOLD}). Proceeding to submission..."
        )

        # Combine Training and Validation data
        X_full = np.concatenate([X_train, X_val], axis=0)
        y_full = np.concatenate([y_train, y_val], axis=0)

        test_preds_list = []

        for name in selected_experts:
            print(f"Retraining {name} on full dataset...")

            # Retrieve the trained pipeline from Phase 1
            phase1_model = trained_models[name]

            # Convert to fixed pipeline (fixes hyperparameters like C for LogisticRegression)
            final_model = models.get_fixed_pipeline(phase1_model)

            # Fit on combined data
            final_model.fit(X_full, y_full)

            # Predict on Test set
            p = final_model.predict_proba(X_test)
            test_preds_list.append(p)

        # Average the predictions from the retrained ensemble
        final_test_probs = np.mean(test_preds_list, axis=0)

        # Save submission
        utils.save_submission(
            test_ids,
            classes,
            final_test_probs,
            output_path="./submission/submission.csv",
        )

    else:
        print(
            f"\nValidation score ({best_val_score}) does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
