import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from library import config
from library import data_loader
from library.feature_pipelines import StreamA_Pipeline, StreamB_Pipeline
from library.trainers import train_random_forest, train_dual_branch_mlp
from library.neural_arch import set_seed


def run_failure_analysis(y_true, y_pred, val_df):
    """
    Performs failure analysis by correlating prediction error with input features.
    """
    print("\n=== Failure Analysis ===")

    # Calculate Error (Absolute difference)
    # y_true is 0 or 1, y_pred is probability [0, 1]
    errors = np.abs(y_true - y_pred)

    # Select numerical columns for correlation
    numeric_cols = val_df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove target if present
    if config.TARGET_COL in numeric_cols:
        numeric_cols.remove(config.TARGET_COL)

    correlations = {}
    for col in numeric_cols:
        if col in val_df.columns:
            # Handle NaNs just in case, though pipeline should have imputed
            feat_values = val_df[col].fillna(0).values
            if len(np.unique(feat_values)) > 1:  # Skip constant columns
                corr = np.corrcoef(errors, feat_values)[0, 1]
                if not np.isnan(corr):
                    correlations[col] = corr

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Prediction Error:")
    for name, val in sorted_corr[:5]:
        print(f"{name}: {val:.4f}")

    return sorted_corr


def main():
    # 1. Setup
    set_seed(config.RANDOM_STATE)
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    print("Starting Hybrid Ensemble Pipeline...")

    # 2. Execute Pipelines
    # Stream A: TF-IDF + Raw Metadata -> Sparse Matrix
    print("\n--- Stream A Pipeline ---")
    pipeline_a = StreamA_Pipeline()
    # Returns: ((X_train, y_train), (X_val, y_val), (X_test, ids_test))
    data_a_train, data_a_val, data_a_test = pipeline_a.run(load_cached_data=True)

    # Stream B: Embeddings + Scaled Metadata -> Dense Arrays
    print("\n--- Stream B Pipeline ---")
    pipeline_b = StreamB_Pipeline()
    # Returns: ((X_sem_tr, X_meta_tr, y_tr), (X_sem_val, X_meta_val, y_val), (X_sem_te, X_meta_te, ids_te))
    data_b_train, data_b_val, data_b_test = pipeline_b.run(load_cached_data=True)

    # 3. Training
    print("\n--- Model Training ---")

    # Train Random Forest (Stream A)
    print("Training Stream A (Random Forest)...")
    rf_model = train_random_forest(data_a_train, data_a_val)

    # Train Dual-Branch MLP (Stream B)
    print("Training Stream B (Dual-Branch MLP)...")
    mlp_trainer = train_dual_branch_mlp(data_b_train, data_b_val)

    # 4. Validation Inference & Ensemble
    print("\n--- Validation & Evaluation ---")

    # RF Predictions (Probabilities for class 1)
    # data_a_val[0] is X_val
    rf_val_probs = rf_model.predict_proba(data_a_val[0])[:, 1]

    # MLP Predictions
    # data_b_val is (X_sem_val, X_meta_val, y_val)
    # predict expects (X_sem, X_meta)
    mlp_val_probs = mlp_trainer.predict((data_b_val[0], data_b_val[1]))

    # Weighted Ensemble
    w_rf = config.ENSEMBLE_WEIGHTS["rf"]
    w_mlp = config.ENSEMBLE_WEIGHTS["mlp"]

    val_preds = (w_rf * rf_val_probs) + (w_mlp * mlp_val_probs)

    # Metric
    y_val = data_a_val[1]  # Targets are consistent across pipelines
    final_auc = roc_auc_score(y_val, val_preds)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_auc}")

    # 5. Failure Analysis
    # Load validation dataframe to map errors to features
    # We use the data_loader to get the dataframe corresponding to the validation set
    _, val_df, _ = data_loader.load_and_clean_data(load_cached_data=True)
    run_failure_analysis(y_val, val_preds, val_df)

    # 6. Submission
    threshold = 0.6591676222161211
    if final_auc > threshold:
        print(
            f"\nValidation metric ({final_auc}) > Threshold ({threshold}). Generating submission..."
        )

        # Test Inference
        # Stream A Test Data: data_a_test -> (X_test, ids_test)
        rf_test_probs = rf_model.predict_proba(data_a_test[0])[:, 1]

        # Stream B Test Data: data_b_test -> (X_sem_test, X_meta_test, ids_test)
        mlp_test_probs = mlp_trainer.predict((data_b_test[0], data_b_test[1]))

        # Ensemble
        test_preds = (w_rf * rf_test_probs) + (w_mlp * mlp_test_probs)

        # Create DataFrame
        test_ids = data_a_test[1]  # IDs are consistent
        submission_df = pd.DataFrame(
            {"request_id": test_ids, "requester_received_pizza": test_preds}
        )

        # Save
        submission_df.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation metric ({final_auc}) <= Threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
