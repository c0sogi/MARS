import os
import sys
import numpy as np
import pandas as pd
import torch
import random
from sklearn.metrics import roc_auc_score

# Import provided library modules
from library import config
from library import utils
from library import feature_factory
from library import rf_model
from library import mlp_model


# =============================================================================
# CONFIGURATION & SEEDING
# =============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run():
    print("Initializing Hybrid Ensemble Pipeline...")
    set_seed(config.RANDOM_STATE)

    # Define validation threshold
    VALIDATION_THRESHOLD = 0.6959737721862433

    # =============================================================================
    # 1. FEATURE GENERATION
    # =============================================================================
    print("\n[Step 1] Generating Features...")
    # Load and process data using the factory
    # This handles LSA, TF-IDF, SBERT, and Metadata generation
    data = feature_factory.create_features(load_cached_data=True)

    # =============================================================================
    # 2. MODEL TRAINING: STREAM A (RANDOM FOREST)
    # =============================================================================
    print("\n[Step 2] Training Stream A: Latent Semantic Random Forest...")
    rf = rf_model.LatentSemanticRF()

    rf.train(
        data["rf"]["train"],
        data["targets"]["train"],
        data["rf"]["val"],
        data["targets"]["val"],
    )

    # Inference on Validation
    print("Generating RF validation predictions...")
    rf_val_preds = rf.predict(data["rf"]["val"])

    # =============================================================================
    # 3. MODEL TRAINING: STREAM B (RESIDUAL-ATTENTION MLP)
    # =============================================================================
    print("\n[Step 3] Training Stream B: Residual-Attention MLP...")

    # Initialize Trainer
    mlp_trainer = mlp_model.MLPTrainer(data["dims"])

    # Train
    mlp_trainer.train(
        data["mlp"]["train"],
        data["targets"]["train"],
        data["mlp"]["val"],
        data["targets"]["val"],
    )

    # Inference on Validation
    print("Generating MLP validation predictions...")
    mlp_val_preds = mlp_trainer.predict(data["mlp"]["val"])

    # =============================================================================
    # 4. ENSEMBLE & EVALUATION
    # =============================================================================
    print("\n[Step 4] Ensembling and Evaluation...")

    # Weighted Average Ensemble
    w_rf = config.ENSEMBLE_WEIGHTS["rf"]
    w_mlp = config.ENSEMBLE_WEIGHTS["mlp"]

    ensemble_val_preds = (w_rf * rf_val_preds) + (w_mlp * mlp_val_preds)

    # Calculate Metric
    val_targets = data["targets"]["val"]
    final_metric = roc_auc_score(val_targets, ensemble_val_preds)

    print(f"Final Validation Metric: {final_metric}")

    # =============================================================================
    # 5. FAILURE ANALYSIS
    # =============================================================================
    print("\n[Step 5] Performing Failure Analysis...")

    # Load raw validation dataframe to access metadata columns for correlation
    _, df_val, _ = utils.load_data(load_cached_data=True)

    # Calculate absolute error
    # Note: For AUC, raw probability error isn't perfect, but it shows where model is "confused"
    # We use |y_true - y_pred|
    errors = np.abs(val_targets - ensemble_val_preds)

    # Select numerical columns for correlation
    numeric_cols = df_val.select_dtypes(include=[np.number]).columns.tolist()
    # Remove target if present
    if "requester_received_pizza" in numeric_cols:
        numeric_cols.remove("requester_received_pizza")

    correlations = {}
    for col in numeric_cols:
        # Handle NaNs in features by filling with mean for correlation check
        feat_values = df_val[col].fillna(df_val[col].mean())
        if len(feat_values.unique()) > 1:
            corr = np.corrcoef(errors, feat_values)[0, 1]
            if not np.isnan(corr):
                correlations[col] = corr

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error Magnitude:")
    for name, val in sorted_corr[:5]:
        print(f"  {name}: {val:.4f}")

    # =============================================================================
    # 6. SUBMISSION
    # =============================================================================
    if final_metric > VALIDATION_THRESHOLD:
        print(
            f"\n[Step 6] Metric {final_metric} > Threshold {VALIDATION_THRESHOLD}. Generating Submission..."
        )

        # Load Test Dataframe for IDs
        _, _, df_test = utils.load_data(load_cached_data=True)

        # Predict RF
        rf_test_preds = rf.predict(data["rf"]["test"])

        # Predict MLP
        mlp_test_preds = mlp_trainer.predict(data["mlp"]["test"])

        # Ensemble
        ensemble_test_preds = (w_rf * rf_test_preds) + (w_mlp * mlp_test_preds)

        # Create Submission DataFrame
        submission = pd.DataFrame(
            {
                "request_id": df_test["request_id"],
                "requester_received_pizza": ensemble_test_preds,
            }
        )

        # Save
        os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
        submission.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")

    else:
        print(
            f"\n[Step 6] Metric {final_metric} <= Threshold {VALIDATION_THRESHOLD}. Skipping Submission."
        )


if __name__ == "__main__":
    run()
