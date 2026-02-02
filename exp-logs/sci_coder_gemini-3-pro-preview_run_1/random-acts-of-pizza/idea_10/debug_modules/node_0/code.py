import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Add the current directory to path to ensure library imports work
sys.path.append(os.getcwd())

# Import provided library modules
from library import config, utils, data_loader, feature_engineering, models


def run_demo():
    print("=== Starting Demonstration Script ===")

    # -------------------------------------------------------------------------
    # 1. CONFIGURATION OVERRIDE FOR SPEED
    # -------------------------------------------------------------------------
    print("\n[Step 1] Overriding configuration for rapid execution...")

    # Use a very small subset of data for demonstration
    config.DEBUG_SAMPLE_SIZE = 50

    # Reduce Random Forest complexity
    config.RF_ESTIMATORS = 10
    config.RF_MAX_DEPTH = 5
    config.RF_N_JOBS = 2

    # Reduce MLP training duration
    config.EPOCHS = 1
    config.BATCH_SIZE = 8
    config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Reduce TF-IDF features to keep sparse matrices small
    config.TFIDF_MAX_FEATURES = 100

    # Set seed for reproducibility
    utils.set_seed(config.SEED)
    print("Configuration updated.")

    # -------------------------------------------------------------------------
    # 2. FEATURE ENGINEERING PIPELINE
    # -------------------------------------------------------------------------
    print("\n[Step 2] Running Feature Engineering Pipeline...")

    # Instantiate pipeline
    pipeline = feature_engineering.FeaturePipeline()

    # Run pipeline (force scratch generation to demo logic)
    # This loads data, computes embeddings, TF-IDF, and custom metadata features
    train_data, val_data, test_data = pipeline.run(load_cached_data=False)

    # Verification of Data Structures
    print("Verifying feature structures...")

    # Check dictionary keys
    required_keys = ["y", "ids", "stream_a", "stream_b"]
    for key in required_keys:
        assert key in train_data, f"Missing key {key} in train_data"
        assert key in val_data, f"Missing key {key} in val_data"

    # Check Stream A (RF) features
    assert "X_tfidf" in train_data["stream_a"]
    assert "X_meta" in train_data["stream_a"]
    assert train_data["stream_a"]["X_tfidf"].shape[0] == config.DEBUG_SAMPLE_SIZE

    # Check Stream B (MLP) features
    assert "X_request_emb" in train_data["stream_b"]
    assert "X_history_emb" in train_data["stream_b"]
    assert train_data["stream_b"]["X_request_emb"].shape[0] == config.DEBUG_SAMPLE_SIZE

    print(f"Data loaded successfully. Train size: {len(train_data['y'])}")

    # -------------------------------------------------------------------------
    # 3. STREAM A: RANDOM FOREST
    # -------------------------------------------------------------------------
    print("\n[Step 3] Training Stream A (Random Forest)...")

    rf_model = models.StreamARandomForest()

    # Train
    rf_model.train(
        train_data["stream_a"]["X_tfidf"],
        train_data["stream_a"]["X_meta"],
        train_data["y"],
    )

    # Predict
    rf_preds_val = rf_model.predict_proba(
        val_data["stream_a"]["X_tfidf"], val_data["stream_a"]["X_meta"]
    )

    # Verify predictions
    assert len(rf_preds_val) == len(val_data["y"])
    assert np.all(
        (rf_preds_val >= 0) & (rf_preds_val <= 1)
    ), "RF predictions out of bounds"

    rf_auc = utils.compute_score(val_data["y"], rf_preds_val)
    print(f"Stream A Validation AUC: {rf_auc:.4f}")

    # -------------------------------------------------------------------------
    # 4. STREAM B: CONTEXT-GATED MLP
    # -------------------------------------------------------------------------
    print("\n[Step 4] Training Stream B (Context-Gated MLP)...")

    # Determine metadata dimension dynamically
    meta_dim = train_data["stream_b"]["X_meta"].shape[1]
    print(f"MLP Metadata Dimension: {meta_dim}")

    mlp_trainer = models.MLPTrainer(meta_dim=meta_dim)

    # Fit model (runs for 1 epoch as configured)
    mlp_trainer.fit(train_data, val_data)

    # Predict
    mlp_preds_val = mlp_trainer.predict_proba(val_data)

    # Verify predictions
    assert len(mlp_preds_val) == len(val_data["y"])
    assert np.all(
        (mlp_preds_val >= 0) & (mlp_preds_val <= 1)
    ), "MLP predictions out of bounds"

    mlp_auc = utils.compute_score(val_data["y"], mlp_preds_val)
    print(f"Stream B Validation AUC: {mlp_auc:.4f}")

    # -------------------------------------------------------------------------
    # 5. ENSEMBLING & SUBMISSION GENERATION
    # -------------------------------------------------------------------------
    print("\n[Step 5] Ensembling and Generating Submission...")

    # Generate Test Predictions
    print("Predicting on Test Set...")

    # Stream A Test Preds
    rf_preds_test = rf_model.predict_proba(
        test_data["stream_a"]["X_tfidf"], test_data["stream_a"]["X_meta"]
    )

    # Stream B Test Preds
    mlp_preds_test = mlp_trainer.predict_proba(test_data)

    # Weighted Average
    w_rf, w_mlp = config.ENSEMBLE_WEIGHTS
    final_preds_test = (w_rf * rf_preds_test) + (w_mlp * mlp_preds_test)

    # Create Submission DataFrame
    submission_df = pd.DataFrame(
        {"request_id": test_data["ids"], "requester_received_pizza": final_preds_test}
    )

    # Verification
    assert len(submission_df) == config.DEBUG_SAMPLE_SIZE
    assert submission_df["request_id"].nunique() == len(submission_df)

    # Save (mock save to working dir)
    output_path = os.path.join(config.WORKING_DIR, "demo_submission.csv")
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print("Head of submission:")
    print(submission_df.head())

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Run the demo
    run_demo()
