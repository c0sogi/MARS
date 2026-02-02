import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder

from library.config import LDA_SHRINKAGE_PARAMS, DTYPE
from library.utils import set_seed, compute_log_loss, save_submission
from library.data_handler import get_data, MORPH_COLS
from library.pipeline_definitions import (
    get_global_pipeline,
    get_global_rotational_pipeline,
    get_stratified_rotational_pipeline,
    get_morphometric_pipeline,
)
from library.ensemble_strategy import GreedySelector, aggregate_predictions


def main():
    # 1. Setup
    set_seed()

    # 2. Load Data
    # load_cached_data=True to use the pre-processed parquet files if available
    X_train_dict, y_train, X_val_dict, y_val, X_test_dict, test_ids, classes = get_data(
        load_cached_data=True
    )

    # 3. Train Expert Library
    # We will train every combination of (Group, Shrinkage) on the Training set
    # and collect predictions on the Validation set.

    library_preds_val = {}
    library_configs = (
        {}
    )  # To store (pipeline_func, view_name, shrinkage) for retraining

    # Define the mapping of Group Name -> (Pipeline Generator Function, Input View Name)
    # Group A: Global Marginal Anchors -> Global View
    # Group B: Global Rotational Experts -> Global View
    # Group C: Stratified Rotational Experts -> Global View (Internal splitting)
    # Group D: Physical Polynomial Experts -> Morphometrics View
    expert_groups = {
        "GroupA": (get_global_pipeline, "Global"),
        "GroupB": (get_global_rotational_pipeline, "Global"),
        "GroupC": (get_stratified_rotational_pipeline, "Global"),
        "GroupD": (get_morphometric_pipeline, "Morphometrics"),
    }

    print(
        f"Training Expert Library with {len(expert_groups) * len(LDA_SHRINKAGE_PARAMS)} candidates..."
    )

    for shrinkage in LDA_SHRINKAGE_PARAMS:
        # Convert shrinkage to string for dictionary keys if needed, but we keep it raw for the function
        shrinkage_key = str(shrinkage)

        for group_name, (pipeline_gen, view_name) in expert_groups.items():
            model_id = f"{group_name}_{shrinkage_key}"

            # Initialize Pipeline
            pipeline = pipeline_gen(shrinkage)

            # Get specific view data
            X_view_train = X_train_dict[view_name]
            X_view_val = X_val_dict[view_name]

            try:
                # Fit on Train
                pipeline.fit(X_view_train, y_train)

                # Predict on Val
                preds_val = pipeline.predict_proba(X_view_val)

                # Store
                library_preds_val[model_id] = preds_val
                library_configs[model_id] = {
                    "pipeline_gen": pipeline_gen,
                    "view_name": view_name,
                    "shrinkage": shrinkage,
                }
            except Exception as e:
                # In case of numerical instability or singular matrices (unlikely with shrinkage)
                print(f"Warning: Model {model_id} failed to train. Error: {e}")

    # 4. Ensemble Selection
    # Use Greedy Forward Selection to find optimal weights based on Val performance
    selector = GreedySelector(n_iterations=100, random_state=42)
    selector.fit(library_preds_val, y_val, classes)
    best_weights = selector.get_best_weights()

    print(f"Selected Experts: {best_weights}")

    # 5. Validation Assessment
    # Compute final metric on validation set
    final_val_preds = aggregate_predictions(library_preds_val, best_weights)
    val_loss = compute_log_loss(y_val, final_val_preds, classes=classes)

    # REQUIRED PRINT FORMAT
    print(f"Final Validation Metric: {val_loss}")

    # 6. Failure Analysis
    print("\nFailure Analysis (Correlation of Error with Morphometric Features):")

    # Calculate per-sample log loss
    # We need to index the probability of the true class
    class_map = {c: i for i, c in enumerate(classes)}
    y_val_indices = np.array([class_map[label] for label in y_val])

    # Clip predictions to avoid log(0) - consistent with metric
    epsilon = 1e-15
    preds_clipped = np.clip(final_val_preds, epsilon, 1 - epsilon)

    # Gather prob of true class for each sample
    probs_true = preds_clipped[np.arange(len(y_val)), y_val_indices]
    sample_losses = -np.log(probs_true)

    # Get Morphometric features for Val set
    X_morph_val = X_val_dict["Morphometrics"]

    # Compute correlations
    for i, col_name in enumerate(MORPH_COLS):
        feature_values = X_morph_val[:, i]
        # Check for constant features to avoid warning
        if np.std(feature_values) == 0:
            corr = 0.0
        else:
            corr, _ = pearsonr(sample_losses, feature_values)
        print(f"  {col_name}: {corr:.6f}")

    # 7. Final Retraining & Submission
    # We proceed to generate submission regardless of the specific threshold in the prompt
    # assuming the goal is to submit the best attempt.

    print("\nRetraining selected experts on combined Train+Val set...")

    test_preds_dict = {}

    for model_id, weight in best_weights.items():
        config = library_configs[model_id]
        pipeline_gen = config["pipeline_gen"]
        view_name = config["view_name"]
        shrinkage = config["shrinkage"]

        # Combine Train and Val data for this view
        X_full = np.vstack([X_train_dict[view_name], X_val_dict[view_name]])
        y_full = np.concatenate([y_train, y_val])

        # Re-initialize and Fit
        pipeline = pipeline_gen(shrinkage)
        pipeline.fit(X_full, y_full)

        # Predict on Test
        X_view_test = X_test_dict[view_name]
        preds_test = pipeline.predict_proba(X_view_test)

        test_preds_dict[model_id] = preds_test

    # Aggregate Test Predictions
    final_test_preds = aggregate_predictions(test_preds_dict, best_weights)

    # Save Submission
    save_submission(test_ids, classes, final_test_preds)


if __name__ == "__main__":
    main()
