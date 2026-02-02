import os
import sys
import numpy as np
import pandas as pd
import torch

# Import library modules
from library import config
from library import data_loader
from library import feature_engineering
from library import text_processing
from library import model_rf
from library import model_mlp

if __name__ == "__main__":
    print("=== Starting Random Acts of Pizza Solution Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration & Reproducibility
    # -------------------------------------------------------------------------
    print("\n[1] Setting up configuration and seeds...")

    # Set seeds for reproducibility
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.RANDOM_SEED)

    # Override hyperparameters for fast demonstration execution
    config.RF_PARAMS["n_estimators"] = 20  # Reduced from 500
    config.RF_PARAMS["n_jobs"] = 2  # Limit parallel jobs
    config.MLP_TRAIN_PARAMS["epochs"] = 2  # Reduced from 50
    config.MLP_TRAIN_PARAMS["batch_size"] = 32
    config.TFIDF_MAX_FEATURES = 500  # Reduced from 5000

    # Ensure working directories exist
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\n[2] Loading Tabular Data...")

    # Force load from source to demonstrate loader logic (ignoring cache)
    df_train, df_val, df_test = data_loader.load_tabular_data(load_cached_data=False)

    # Verify data integrity
    print(f"    Train shape: {df_train.shape}")
    print(f"    Val shape:   {df_val.shape}")
    print(f"    Test shape:  {df_test.shape}")

    assert len(df_train) > 0, "Training data should not be empty."
    assert (
        config.TARGET_COL in df_train.columns
    ), f"Target '{config.TARGET_COL}' missing."
    # Check if list parsing worked (history column should be a list, not a string)
    assert isinstance(
        df_train[config.HISTORY_COL].iloc[0], list
    ), "History column parsing failed."

    # -------------------------------------------------------------------------
    # 3. Feature Engineering (Metadata + Target Encoding)
    # -------------------------------------------------------------------------
    print("\n[3] Generating Engineered Features...")

    # Generate features and save to cache.
    # We use load_cached_data=False to force execution of the engineering logic.
    X_train, X_val, X_test = feature_engineering.generate_features(
        load_cached_data=False
    )

    print(f"    Feature Matrix Shape: {X_train.shape}")

    # Verify specific engineered columns exist
    assert (
        "meta_text_len_chars" in X_train.columns
    ), "Metadata feature 'meta_text_len_chars' missing."
    assert (
        "te_generosity_mean" in X_train.columns
    ), "Target Encoding feature 'te_generosity_mean' missing."
    assert X_train.shape[0] == len(df_train), "Feature matrix row count mismatch."

    # -------------------------------------------------------------------------
    # 4. Text Processing (TF-IDF & SBERT)
    # -------------------------------------------------------------------------
    print("\n[4] Generating Text Features...")

    # A. TF-IDF
    print("    -> Running TF-IDF Pipeline...")
    tfidf_pipeline = text_processing.TfidfPipeline(
        max_features=config.TFIDF_MAX_FEATURES
    )
    tfidf_feats = tfidf_pipeline.run(df_train, df_val, df_test, load_cached_data=False)

    assert (
        tfidf_feats["train"].shape[1] == config.TFIDF_MAX_FEATURES
    ), "TF-IDF feature dimension mismatch."

    # B. SBERT
    print("    -> Running SBERT Encoder...")
    sbert_encoder = text_processing.SbertEncoder(batch_size=32)
    # Generate and cache embeddings
    req_emb = sbert_encoder.generate_request_embeddings(
        df_train, df_val, df_test, load_cached_data=False
    )
    hist_emb = sbert_encoder.generate_history_embeddings(
        df_train, df_val, df_test, load_cached_data=False
    )

    assert (
        req_emb["train"].shape[1] == 384
    ), "SBERT embedding dimension incorrect (expected 384)."

    # -------------------------------------------------------------------------
    # 5. Model Stream A: Random Forest
    # -------------------------------------------------------------------------
    print("\n[5] Running Random Forest Stream...")

    rf_stream = model_rf.RandomForestStream()
    # We set load_cached_data=True here to utilize the features we just generated/cached
    rf_val_probs, rf_test_probs, rf_model = rf_stream.run(load_cached_data=True)

    print(f"    RF Test Predictions: {rf_test_probs[:5]}")
    assert len(rf_test_probs) == len(df_test), "RF test predictions length mismatch."

    # -------------------------------------------------------------------------
    # 6. Model Stream B: MLP
    # -------------------------------------------------------------------------
    print("\n[6] Running MLP Stream...")

    mlp_stream = model_mlp.MLPStream()
    # Uses cached SBERT embeddings and metadata
    mlp_val_probs, mlp_test_probs, mlp_model = mlp_stream.run(load_cached_data=True)

    print(f"    MLP Test Predictions: {mlp_test_probs[:5]}")
    assert len(mlp_test_probs) == len(df_test), "MLP test predictions length mismatch."

    # -------------------------------------------------------------------------
    # 7. Ensemble & Submission
    # -------------------------------------------------------------------------
    print("\n[7] Ensembling and Generating Submission...")

    w_rf, w_mlp = config.ENSEMBLE_WEIGHTS
    print(f"    Weights -> RF: {w_rf}, MLP: {w_mlp}")

    # Weighted Average
    final_test_probs = (w_rf * rf_test_probs) + (w_mlp * mlp_test_probs)

    # Create Submission DataFrame
    submission = pd.DataFrame(
        {
            "request_id": df_test["request_id"],
            "requester_received_pizza": final_test_probs,
        }
    )

    # Save to disk
    submission.to_csv(config.SUBMISSION_PATH, index=False)

    print(f"    Submission saved to: {config.SUBMISSION_PATH}")
    print("    First 5 rows:")
    print(submission.head())

    print("\n=== Demo Completed Successfully ===")
