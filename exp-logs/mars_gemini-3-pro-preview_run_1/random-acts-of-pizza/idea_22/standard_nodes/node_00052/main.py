import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import torch

# Ensure library is in path
sys.path.append(".")

from library.config import Config
from library.data_loader import load_dataset
from library.feature_engineering import FeatureEngineer
from library.neural_net import NeuralNetTrainer


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override Config for Fast Baseline Execution
    Config.RF_N_ESTIMATORS = 100
    Config.MLP_EPOCHS = 20
    Config.MLP_PATIENCE = 5

    # Set Reproducibility Seeds
    np.random.seed(Config.RANDOM_SEED)
    torch.manual_seed(Config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.RANDOM_SEED)

    print("Configuration set for fast baseline execution.")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("Loading datasets...")
    train_df, val_df, test_df = load_dataset(load_cached_data=True)

    # ==========================================
    # 3. Feature Engineering
    # ==========================================
    print("Initializing Feature Engineering...")
    fe = FeatureEngineer()

    # Prepare Random Forest Inputs (Stream A)
    print("Preparing Random Forest Inputs...")
    (X_train_rf, y_train_rf), (X_val_rf, y_val_rf), X_test_rf = fe.prepare_rf_inputs(
        train_df, val_df, test_df, load_cached_data=True
    )

    # Prepare MLP Inputs (Stream B)
    print("Preparing MLP Inputs...")
    train_data_mlp, val_data_mlp, test_data_mlp = fe.prepare_mlp_inputs(
        train_df, val_df, test_df, load_cached_data=True
    )

    # ==========================================
    # 4. Model Training
    # ==========================================

    # --- Stream A: Random Forest ---
    print("Training Random Forest (Stream A)...")
    rf_model = RandomForestClassifier(
        n_estimators=Config.RF_N_ESTIMATORS,
        class_weight=Config.RF_CLASS_WEIGHT,
        max_depth=Config.RF_MAX_DEPTH,
        min_samples_leaf=Config.RF_MIN_SAMPLES_LEAF,
        n_jobs=Config.RF_N_JOBS,
        random_state=Config.RANDOM_SEED,
    )
    rf_model.fit(X_train_rf, y_train_rf)

    # Validation Predictions (RF)
    rf_val_preds = rf_model.predict_proba(X_val_rf)[:, 1]

    # --- Stream B: Gated Attention MLP ---
    print("Training Gated Attention MLP (Stream B)...")
    meta_dim = train_data_mlp["metadata"].shape[1]
    trainer = NeuralNetTrainer(input_dims={"metadata": meta_dim})

    # Train
    trainer.train(train_data_mlp, val_data_mlp)

    # Validation Predictions (MLP)
    # Hack: Remove 'y' from validation data to treat it as inference data for raw probabilities
    val_data_inference = val_data_mlp.copy()
    if "y" in val_data_inference:
        del val_data_inference["y"]

    mlp_val_preds = trainer.predict(val_data_inference)

    # ==========================================
    # 5. Validation & Ensembling
    # ==========================================
    print("Ensembling predictions...")

    # Simple Weighted Average
    final_val_preds = 0.5 * rf_val_preds + 0.5 * mlp_val_preds

    # Calculate Metric
    val_auc = roc_auc_score(y_val_rf, final_val_preds)
    print(f"Final Validation Metric: {val_auc}")

    # ==========================================
    # 6. Failure Analysis
    # ==========================================
    print("\n=== Failure Analysis ===")
    # Load interpretable metadata for correlation analysis
    _, meta_val_df, _ = fe.extract_metadata(
        train_df, val_df, test_df, load_cached_data=True
    )

    # Calculate Error Magnitude
    # y_val_rf is int (0/1), preds are floats [0,1]
    errors = np.abs(y_val_rf - final_val_preds)

    # Create analysis dataframe
    analysis_df = meta_val_df.copy()
    analysis_df["error_magnitude"] = errors

    # Compute correlations
    correlations = analysis_df.corrwith(analysis_df["error_magnitude"]).sort_values(
        ascending=False
    )

    # Remove the error column itself from correlations
    correlations = correlations.drop("error_magnitude", errors="ignore")

    print("Top 5 Features correlated with higher error:")
    print(correlations.head(5))

    print("\nTop 5 Features correlated with lower error:")
    print(correlations.tail(5))

    # ==========================================
    # 7. Submission Generation
    # ==========================================
    threshold = 0.6959737721862433

    if val_auc > threshold:
        print(
            f"\nValidation metric {val_auc} exceeds threshold {threshold}. Generating submission..."
        )

        # RF Inference
        rf_test_preds = rf_model.predict_proba(X_test_rf)[:, 1]

        # MLP Inference
        mlp_test_preds = trainer.predict(test_data_mlp)

        # Ensemble
        final_test_preds = 0.5 * rf_test_preds + 0.5 * mlp_test_preds

        # Create Submission DataFrame
        submission_df = pd.DataFrame(
            {
                "request_id": test_df["request_id"],
                "requester_received_pizza": final_test_preds,
            }
        )

        # Save
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation metric {val_auc} did not exceed threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
