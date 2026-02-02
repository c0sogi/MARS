import sys
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import roc_auc_score

# Import provided library modules
from library import (
    config,
    utils,
    feature_engineering,
    train_eval,
    neural_net,
    data_loader,
)


def main():
    # 1. Setup and Configuration
    # Set random seeds for reproducibility
    utils.set_seed()

    # Restore full training configuration (Cite solution_lesson_node_00016)
    # Neural networks on small data need sufficient warm-up and patience.
    # We use the defaults from config.py (50 epochs, 15 patience).

    # 2. Data Loading and Feature Engineering
    print("Initializing Feature Processor...")
    processor = feature_engineering.FeatureProcessor()

    # Process data (loads from cache if available)
    data = processor.process_data(load_cached_data=True)

    train_data = data["train"]
    val_data = data["val"]
    test_data = data["test"]

    # 3. Model Training
    # Train Learner A: Random Forest
    print("\n--- Training Learner A (Random Forest) ---")
    rf_model = train_eval.train_rf(train_data, val_data)

    # Train Learner B: Dual-Branch MLP
    print("\n--- Training Learner B (Dual-Branch MLP) ---")
    mlp_model = train_eval.train_mlp_wrapper(train_data, val_data)

    # 4. Validation and Ensemble Evaluation
    print("\n--- Validating Ensemble ---")

    # Generate RF Validation Probabilities
    # We need to manually construct the sparse matrix here as done in train_rf
    X_val_rf = sparse.hstack([val_data["dense"], val_data["tfidf"]])
    rf_val_probs = rf_model.predict_proba(X_val_rf)[:, 1]

    # Generate MLP Validation Probabilities
    mlp_val_probs = neural_net.predict_model(mlp_model, val_data)

    # Calculate Weighted Ensemble
    w_rf = config.ENSEMBLE_WEIGHTS["rf"]
    w_mlp = config.ENSEMBLE_WEIGHTS["mlp"]

    ensemble_val_probs = (w_rf * rf_val_probs) + (w_mlp * mlp_val_probs)

    # Calculate Metric
    val_auc = roc_auc_score(val_data["y"], ensemble_val_probs)

    # Print required metric format
    print(f"Final Validation Metric: {val_auc}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Load raw metadata to get feature names and original values
    _, val_df, _ = data_loader.load_metadata_splits(load_cached_data=True)

    # Ensure alignment (data_loader and process_data use same split logic/files)
    if len(val_df) != len(ensemble_val_probs):
        print(
            "Warning: Validation dataframe length mismatch. Skipping detailed failure analysis."
        )
    else:
        # Calculate Error
        val_df["pred_prob"] = ensemble_val_probs
        val_df["target"] = val_data["y"]
        val_df["error"] = np.abs(val_df["target"] - val_df["pred_prob"])

        # Calculate correlations between numerical features and error
        numeric_cols = val_df.select_dtypes(include=[np.number]).columns
        correlations = []

        exclude_cols = ["pred_prob", "target", "error", "requester_received_pizza"]

        for col in numeric_cols:
            if col not in exclude_cols:
                # Handle potential NaNs in raw data
                if val_df[col].isnull().any():
                    series = val_df[col].fillna(0)
                else:
                    series = val_df[col]

                # Compute correlation
                try:
                    corr = series.corr(val_df["error"])
                    if not np.isnan(corr):
                        correlations.append((col, corr))
                except Exception:
                    continue

        # Sort by absolute correlation magnitude
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        print("Top 10 features correlated with prediction error (Failure Analysis):")
        for name, corr in correlations[:10]:
            print(f"{name:<50}: {corr:.4f}")

    # 6. Conditional Submission
    THRESHOLD = 0.6789999838498684

    if val_auc > THRESHOLD:
        print(f"\nValidation metric {val_auc} exceeds threshold {THRESHOLD}.")
        print("Generating submission for test set...")

        # Generate Test Predictions
        test_probs = train_eval.predict_ensemble(rf_model, mlp_model, test_data)

        # Save Submission
        utils.save_submission(test_probs, test_data["ids"])
    else:
        print(f"\nValidation metric {val_auc} does not exceed threshold {THRESHOLD}.")
        print("Submission generation skipped.")


if __name__ == "__main__":
    main()
