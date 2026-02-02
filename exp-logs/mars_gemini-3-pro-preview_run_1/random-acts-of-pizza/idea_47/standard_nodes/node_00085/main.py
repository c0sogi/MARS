import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------
# 1. Configuration adjustments for Fast Baseline
# ---------------------------------------------------------
# We modify the configuration before importing other modules to ensure
# these settings take effect for a faster run.
import library.config as config

config.RF_N_ESTIMATORS = 100  # Reduced from 500 for speed
config.MLP_NUM_EPOCHS = 15  # Reduced from 50 for speed
config.MLP_PATIENCE = 5  # Reduced patience

# ---------------------------------------------------------
# 2. Library Imports
# ---------------------------------------------------------
from library.utils import seed_everything
from library.feature_engineering import FeaturePipeline
from library.train import train_rf, train_mlp
from library.dataset import create_dataloaders
from library.predict import generate_mlp_predictions, ensemble_predictions
from library.models import InteractionRandomForest


def run():
    # Set seeds for reproducibility
    seed_everything()

    print("=== Starting Run Pipeline ===")

    # ---------------------------------------------------------
    # 3. Feature Engineering
    # ---------------------------------------------------------
    print("\n[Step 1] Feature Engineering...")
    # Initialize pipeline and run. This handles caching automatically.
    # We need the output dictionaries to access validation features and labels.
    pipeline = FeaturePipeline()
    rf_out, mlp_out = pipeline.run(load_cached_data=True)

    # ---------------------------------------------------------
    # 4. Model Training
    # ---------------------------------------------------------
    print("\n[Step 2] Training Models...")

    # Stream A: Random Forest
    print("Training Interaction-Enhanced Random Forest...")
    rf_model = train_rf(load_cached_data=True)

    # Stream B: FiLM-Conditioned MLP
    print(f"Training FiLM-Conditioned MLP (Epochs={config.MLP_NUM_EPOCHS})...")
    mlp_model = train_mlp(load_cached_data=True, batch_size=config.MLP_BATCH_SIZE)

    # ---------------------------------------------------------
    # 5. Validation & Evaluation
    # ---------------------------------------------------------
    print("\n[Step 3] Validation & Evaluation...")

    # --- Prepare Validation Data ---
    # RF: Features are already in rf_out
    X_val_rf = rf_out["val_X"]
    y_val = rf_out["val_y"]

    # MLP: Need to create dataloaders to get the validation loader correctly
    # We discard train_loader here as we only need val/test for inference
    _, val_loader, test_loader = create_dataloaders(
        load_cached_data=True, batch_size=config.MLP_BATCH_SIZE
    )

    # --- Inference ---
    print("Generating validation predictions...")

    # RF Predictions
    rf_val_probs = rf_model.predict_proba(X_val_rf)[:, 1]

    # MLP Predictions
    # Note: generate_mlp_predictions handles moving data to GPU and eval mode
    mlp_val_probs = generate_mlp_predictions(mlp_model, val_loader)

    # --- Ensemble ---
    val_preds = ensemble_predictions(rf_val_probs, mlp_val_probs)

    # --- Metric Calculation ---
    val_auc = roc_auc_score(y_val, val_preds)
    print(f"Final Validation Metric: {val_auc}")

    # ---------------------------------------------------------
    # 6. Failure Analysis
    # ---------------------------------------------------------
    print("\n[Step 4] Failure Analysis...")

    # Calculate absolute error magnitude
    errors = np.abs(y_val - val_preds)

    # Load interpretable metadata from the CSV file for correlation analysis
    val_df = pd.read_csv(config.VAL_PATH)
    val_df["prediction_error"] = errors

    # Identify numeric columns for correlation
    numeric_cols = val_df.select_dtypes(include=[np.number]).columns.tolist()

    # Exclude target and the error column itself
    exclude_cols = ["requester_received_pizza", "prediction_error"]
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

    correlations = {}
    for col in numeric_cols:
        # Calculate correlation if the column has variance
        if val_df[col].std() > 1e-6:
            # Simple imputation for correlation calculation
            col_data = val_df[col].fillna(val_df[col].median())
            corr = col_data.corr(val_df["prediction_error"])
            correlations[col] = corr

    # Sort by absolute correlation strength
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top Feature Correlations with Error Magnitude:")
    for name, score in sorted_corr[:5]:
        print(f"  {name}: {score:.6f}")

    # ---------------------------------------------------------
    # 7. Submission
    # ---------------------------------------------------------
    print("\n[Step 5] Submission Generation...")

    THRESHOLD = 0.7135451153926904

    if val_auc > THRESHOLD:
        print(
            f"Validation AUC ({val_auc:.6f}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # --- RF Test Inference ---
        X_test_rf = rf_out["test_X"]
        rf_test_probs = rf_model.predict_proba(X_test_rf)[:, 1]

        # --- MLP Test Inference ---
        mlp_test_probs = generate_mlp_predictions(mlp_model, test_loader)

        # --- Ensemble ---
        test_preds = ensemble_predictions(rf_test_probs, mlp_test_probs)

        # --- Create Submission File ---
        test_df_raw = pd.read_csv(config.TEST_PATH)
        submission_df = pd.DataFrame(
            {
                "request_id": test_df_raw["request_id"],
                "requester_received_pizza": test_preds,
            }
        )

        # Ensure directory exists and save
        os.makedirs(os.path.dirname(config.OUTPUT_SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(config.OUTPUT_SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.OUTPUT_SUBMISSION_PATH}")

        # Print head for verification
        print(submission_df.head())

    else:
        print(
            f"Validation AUC ({val_auc:.6f}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    run()
