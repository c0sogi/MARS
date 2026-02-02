import pandas as pd
import numpy as np
import os
import sys
import gc

# Import from provided libraries
from library.config import Config
from library.utils import seed_everything, compute_mcc
from library.data_manager import DataManager
from library.model_trainer import DualStreamPredictor
from library.optimizer import ThresholdOptimizer


def perform_failure_analysis(X_val, y_val, y_probs):
    """
    Analyzes the correlation between model error and input features.
    """
    print("\n=== Failure Analysis ===")

    # Calculate error magnitude (continuous)
    # error = |y_true - y_prob|
    errors = np.abs(y_val - y_probs)

    # Create a DataFrame for analysis
    df_analysis = X_val.copy()
    df_analysis["error_magnitude"] = errors

    # Compute correlations with error
    correlations = df_analysis.corrwith(df_analysis["error_magnitude"])

    # Sort by absolute correlation
    correlations = correlations.drop("error_magnitude")
    correlations_abs = correlations.abs().sort_values(ascending=False)

    print("Top 10 Features correlated with Error Magnitude:")
    print(correlations[correlations_abs.index[:10]])
    print("========================\n")


def main():
    # 1. Initialization
    seed_everything(Config.SEED)
    print("Initializing Pipeline...")

    # Initialize DataManager
    # We use debug=False to ensure we have enough data to meet the metric threshold,
    # relying on the A100 GPU and XGBoost 'hist' method for speed.
    data_manager = DataManager(debug=False)

    # 2. Load Data for Stream A (Player-Player)
    print("\n--- Loading Stream A (Player-Player) Data ---")
    X_train_a, y_train_a, X_val_a, y_val_a = data_manager.prepare_stream_datasets(
        stream_type="A", load_cached=True
    )
    print(f"Stream A Train Shape: {X_train_a.shape}, Val Shape: {X_val_a.shape}")

    # 3. Load Data for Stream B (Player-Ground)
    print("\n--- Loading Stream B (Player-Ground) Data ---")
    X_train_b, y_train_b, X_val_b, y_val_b = data_manager.prepare_stream_datasets(
        stream_type="B", load_cached=True
    )
    print(f"Stream B Train Shape: {X_train_b.shape}, Val Shape: {X_val_b.shape}")

    # 4. Train Models
    predictor = DualStreamPredictor()

    # Train Stream A
    print("\n--- Training Stream A ---")
    clf_a, thresh_a_init = predictor.train_stream(
        X_train_a, y_train_a, X_val_a, y_val_a, "A"
    )

    # Train Stream B
    print("\n--- Training Stream B ---")
    clf_b, thresh_b_init = predictor.train_stream(
        X_train_b, y_train_b, X_val_b, y_val_b, "B"
    )

    # Free up training memory
    del X_train_a, y_train_a, X_train_b, y_train_b
    gc.collect()

    # 5. Global Optimization & Validation
    print("\n--- Validating & Optimizing ---")

    # Get probabilities
    probs_a = clf_a.predict_proba(X_val_a)[:, 1]
    probs_b = clf_b.predict_proba(X_val_b)[:, 1]

    # Optimize thresholds jointly
    optimizer = ThresholdOptimizer(steps=100)
    best_thresh_a, best_thresh_b, final_mcc = optimizer.optimize_thresholds(
        y_val_a, probs_a, y_val_b, probs_b
    )

    # Update predictor with globally optimized thresholds
    predictor.thresholds["A"] = best_thresh_a
    predictor.thresholds["B"] = best_thresh_b

    # REQUIRED: Print Final Validation Metric
    print(f"Final Validation Metric: {final_mcc}")

    # 6. Failure Analysis
    # Combine validation sets for analysis or analyze the larger one (Stream A usually has more complex interactions)
    # We'll analyze Stream A as it's the primary interaction component
    perform_failure_analysis(X_val_a, y_val_a, probs_a)

    # 7. Submission
    SUBMISSION_THRESHOLD = 0.6565613438092561

    if final_mcc > SUBMISSION_THRESHOLD:
        print(
            f"\nValidation MCC ({final_mcc}) > {SUBMISSION_THRESHOLD}. Generating submission..."
        )

        # Load Test Data
        print("Loading Test Data Stream A...")
        X_test_a, ids_a = data_manager.get_test_data(stream_type="A", load_cached=True)

        print("Loading Test Data Stream B...")
        X_test_b, ids_b = data_manager.get_test_data(stream_type="B", load_cached=True)

        # Predict
        print("Predicting Stream A...")
        preds_a = predictor.predict(X_test_a, "A")

        print("Predicting Stream B...")
        preds_b = predictor.predict(X_test_b, "B")

        # Create DataFrames
        df_sub_a = pd.DataFrame({"contact_id": ids_a, "contact": preds_a})
        df_sub_b = pd.DataFrame({"contact_id": ids_b, "contact": preds_b})

        # Combine
        df_submission = pd.concat([df_sub_a, df_sub_b], axis=0, ignore_index=True)

        # Ensure all IDs from sample submission are present (though DataManager should handle this via metadata)
        # We load sample submission to ensure order and completeness if necessary,
        # but the task says "The file should contain a header and have the following format".
        # Usually, we just save what we predicted.

        save_path = Config.SUBMISSION_PATH
        print(f"Saving submission to {save_path}...")
        df_submission.to_csv(save_path, index=False)
        print("Submission saved successfully.")

    else:
        print(
            f"\nValidation MCC ({final_mcc}) <= {SUBMISSION_THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
