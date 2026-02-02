import os
import sys
import numpy as np
import pandas as pd
import torch
import shutil

# Import library modules
# We import them to verify they load correctly and to patch parameters for speed
import library.config as config
import library.data_loader as data_loader
import library.text_processing as text_processing
import library.feature_engineering as feature_engineering
import library.rf_model as rf_model_module
import library.mlp_model as mlp_model_module
import library.trainer as trainer_module


def run_demo():
    print("=== Starting Library Usage Demo ===")

    # -------------------------------------------------------------------------
    # 1. SETUP & HYPERPARAMETER PATCHING (For Speed)
    # -------------------------------------------------------------------------
    print("Patching hyperparameters for fast execution...")

    # Patch Random Forest parameters
    rf_model_module.RF_PARAMS["n_estimators"] = 10
    rf_model_module.RF_PARAMS["n_jobs"] = 1  # Reduce overhead for small data
    rf_model_module.RF_PARAMS["verbose"] = 0

    # Patch MLP parameters
    mlp_model_module.MLP_PARAMS["epochs"] = 2
    mlp_model_module.MLP_PARAMS["batch_size"] = 8
    mlp_model_module.MLP_PARAMS["hidden_dim"] = 32
    mlp_model_module.MLP_PARAMS["film_dim"] = 16
    mlp_model_module.MLP_PARAMS["patience"] = 1

    # Patch Text Processing parameters to reduce vocabulary size
    text_processing.TEXT_CONFIG["tfidf_max_features"] = 100

    # Ensure reproducibility
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    # -------------------------------------------------------------------------
    # 2. DATA LOADING
    # -------------------------------------------------------------------------
    print("\n[Step 1] Loading Data (Debug Mode)...")
    # Load a small subset (50 samples) to ensure speed.
    # Must set load_cached_data=False to force reloading the subset.
    debug_size = 50
    train_df, val_df, test_df = data_loader.load_dataset(
        load_cached_data=False, debug_size=debug_size
    )

    # Verification
    print(f"Train shape: {train_df.shape}")
    print(f"Val shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")

    assert len(train_df) <= debug_size
    assert len(val_df) <= debug_size
    assert len(test_df) <= debug_size
    assert config.TARGET_COL in train_df.columns
    assert config.LIST_COL in train_df.columns

    # -------------------------------------------------------------------------
    # 3. FEATURE GENERATION (Text & Metadata)
    # -------------------------------------------------------------------------
    print("\n[Step 2] Generating Text Features...")

    # Generate SBERT Embeddings
    # Note: This might download the model if not present, but usually fast for MiniLM
    sbert_data = text_processing.generate_sbert_embeddings(
        train_df, val_df, test_df, load_cached_data=False
    )

    # Verify SBERT output
    assert "train_title" in sbert_data
    assert "train_hist_centroid" in sbert_data
    assert sbert_data["train_title"].shape[0] == len(train_df)

    # Generate TF-IDF Features
    tfidf_data = text_processing.generate_tfidf_features(
        train_df, val_df, test_df, load_cached_data=False
    )

    # Verify TF-IDF output
    assert "train_tfidf" in tfidf_data
    assert tfidf_data["train_tfidf"].shape[0] == len(train_df)

    # -------------------------------------------------------------------------
    # 4. PREPARING MODEL INPUTS
    # -------------------------------------------------------------------------
    print("\n[Step 3] Preparing Model-Specific Features...")

    # Random Forest Features
    rf_features = feature_engineering.prepare_rf_features(
        train_df, val_df, test_df, tfidf_data, sbert_data, load_cached_data=False
    )

    # Verify RF Features
    assert "X_train" in rf_features
    assert "y_train" in rf_features
    assert rf_features["X_train"].shape[0] == len(train_df)

    # MLP Features
    mlp_features = feature_engineering.prepare_mlp_features(
        train_df, val_df, test_df, sbert_data, load_cached_data=False
    )

    # Verify MLP Features
    assert "train_metadata" in mlp_features
    assert "train_target" in mlp_features
    assert mlp_features["train_metadata"].shape[0] == len(train_df)

    # -------------------------------------------------------------------------
    # 5. RANDOM FOREST MODEL (Stream A)
    # -------------------------------------------------------------------------
    print("\n[Step 4] Training Random Forest Model...")

    rf_model = rf_model_module.InteractionRandomForest()

    # Train
    rf_model.train(
        rf_features["X_train"],
        rf_features["y_train"],
        rf_features["X_val"],
        rf_features["y_val"],
    )

    # Predict
    rf_preds_val = rf_model.predict_proba(rf_features["X_val"])
    rf_preds_test = rf_model.predict_proba(rf_features["X_test"])

    # Verify Predictions
    assert len(rf_preds_val) == len(val_df)
    assert np.all((rf_preds_val >= 0) & (rf_preds_val <= 1))

    # Save/Load Check
    rf_model.save()
    assert os.path.exists(rf_model.model_path)
    loaded = rf_model.load()
    assert loaded is True

    # -------------------------------------------------------------------------
    # 6. MLP MODEL (Stream B)
    # -------------------------------------------------------------------------
    print("\n[Step 5] Training MLP Model...")

    mlp_trainer = mlp_model_module.MLPTrainer()

    # Train
    mlp_trainer.train(mlp_features)

    # Predict
    mlp_preds_val = mlp_trainer.predict_proba(mlp_features, split_name="val")
    mlp_preds_test = mlp_trainer.predict_proba(mlp_features, split_name="test")

    # Verify Predictions
    assert len(mlp_preds_val) == len(val_df)
    assert np.all((mlp_preds_val >= 0) & (mlp_preds_val <= 1))

    # Save/Load Check
    assert os.path.exists(mlp_trainer.model_path)
    loaded = mlp_trainer.load()
    assert loaded is True

    # -------------------------------------------------------------------------
    # 7. SUBMISSION GENERATION
    # -------------------------------------------------------------------------
    print("\n[Step 6] Generating Submission...")

    submission_df = trainer_module.generate_submission(
        test_df, rf_preds_test, mlp_preds_test, ensemble_weights=(0.6, 0.4)
    )

    # Verify Submission
    assert len(submission_df) == len(test_df)
    assert config.ID_COL in submission_df.columns
    assert config.TARGET_COL in submission_df.columns
    assert os.path.exists(os.path.join(config.SUBMISSION_DIR, "submission.csv"))

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    # Ensure working directory exists (it should be created by config, but good to be safe)
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    try:
        run_demo()
    except Exception as e:
        print(f"\n!!! Demo Failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
