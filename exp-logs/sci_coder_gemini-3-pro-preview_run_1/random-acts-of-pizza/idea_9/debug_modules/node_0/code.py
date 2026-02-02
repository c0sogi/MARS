import os
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Import components from the provided library
from library.config import Config
from library.utils import set_seed, save_submission
from library.data_loader import DataLoader
from library.feature_engineering import SemanticProcessor, TabularProcessor
from library.model_rf import train_predict_rf
from library.model_mlp import train_mlp, predict_mlp


def run_demo_pipeline():
    print("=== Starting Demo Pipeline Execution ===")

    # --------------------------------------------------------------------------
    # 1. Configuration Overrides
    # --------------------------------------------------------------------------
    # We modify the Config class attributes directly to optimize for a fast demo run.
    print("Configuring environment for rapid execution...")

    Config.DEBUG = True
    Config.DEBUG_SIZE = 60  # Small subset for speed
    Config.BATCH_SIZE = 8  # Small batch size to ensure batches exist with small data
    Config.NUM_EPOCHS = 2  # Minimal epochs
    Config.PATIENCE = 1  # Minimal patience

    # Reduce Random Forest complexity
    Config.RF_PARAMS["n_estimators"] = 10

    # Set a specific working directory for this demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Clean up demo directory if it exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set global seed
    set_seed(Config.SEED)

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    print("\n[Step 1] Loading Data...")
    # We force load_cached_data=False to demonstrate the raw loading and parsing logic
    df_train, df_val, df_test = DataLoader.load_raw_data(load_cached_data=False)

    # Validation
    print(
        f"Train shape: {df_train.shape}, Val shape: {df_val.shape}, Test shape: {df_test.shape}"
    )
    assert len(df_train) == Config.DEBUG_SIZE, "Train set size mismatch for debug mode"
    assert len(df_val) == Config.DEBUG_SIZE, "Val set size mismatch for debug mode"
    assert len(df_test) == Config.DEBUG_SIZE, "Test set size mismatch for debug mode"
    assert (
        Config.SUBREDDIT_LIST_COL in df_train.columns
    ), "Subreddit list column missing"
    assert isinstance(
        df_train.iloc[0][Config.SUBREDDIT_LIST_COL], list
    ), "Subreddit column not parsed as list"

    # --------------------------------------------------------------------------
    # 3. Feature Selection
    # --------------------------------------------------------------------------
    print("\n[Step 2] Selecting Safe Features...")
    safe_cols = DataLoader.filter_leakage_columns(df_train, df_test)

    # Validation
    assert len(safe_cols) > 0, "No safe features identified"
    assert Config.TARGET_COL not in safe_cols, "Target column leaked into features"
    print(f"Selected {len(safe_cols)} safe tabular features.")

    # --------------------------------------------------------------------------
    # 4. Feature Engineering (Semantic - Stream B)
    # --------------------------------------------------------------------------
    print("\n[Step 3] Generating Semantic Features (SBERT)...")
    sem_processor = SemanticProcessor()

    # Process data (this handles text embeddings and subreddit history tensors)
    train_text, train_subs, val_text, val_subs, test_text, test_subs = (
        sem_processor.process_data(df_train, df_val, df_test, load_cached_data=False)
    )

    # Validation
    # Expected text shape: (N, 384)
    # Expected sub shape: (N, MAX_SEQ_LEN, 384)
    assert train_text.shape == (Config.DEBUG_SIZE, Config.SBERT_EMBEDDING_DIM)
    assert train_subs.shape == (
        Config.DEBUG_SIZE,
        Config.MAX_SUBREDDIT_SEQ_LEN,
        Config.SBERT_EMBEDDING_DIM,
    )
    print("Semantic features generated successfully.")

    # --------------------------------------------------------------------------
    # 5. Feature Engineering (Tabular - Stream A & B)
    # --------------------------------------------------------------------------
    print("\n[Step 4] Generating Tabular Features (TF-IDF & Metadata)...")
    tab_processor = TabularProcessor()

    (
        train_tfidf,
        train_meta_a,
        train_meta_b,
        val_tfidf,
        val_meta_a,
        val_meta_b,
        test_tfidf,
        test_meta_a,
        test_meta_b,
    ) = tab_processor.process_data(
        df_train, df_val, df_test, safe_cols, load_cached_data=False
    )

    # Validation
    assert train_tfidf.shape[0] == Config.DEBUG_SIZE
    assert train_meta_a.shape[0] == Config.DEBUG_SIZE
    assert train_meta_b.shape[0] == Config.DEBUG_SIZE
    # Check that Stream B metadata is scaled (mean approx 0, std approx 1)
    # We check one column to be roughly standard normal
    assert (
        np.abs(train_meta_b[:, 0].mean()) < 1.0
    ), "Stream B metadata does not appear centered"
    print("Tabular features generated successfully.")

    # --------------------------------------------------------------------------
    # 6. Model Training: Stream A (Random Forest)
    # --------------------------------------------------------------------------
    print("\n[Step 5] Running Stream A (Random Forest)...")

    # Prepare targets
    train_y = df_train[Config.TARGET_COL].astype(int).values
    val_y = df_val[Config.TARGET_COL].astype(int).values

    val_probs_rf, test_probs_rf, model_rf = train_predict_rf(
        train_tfidf,
        train_meta_a,
        train_y,
        val_tfidf,
        val_meta_a,
        val_y,
        test_tfidf,
        test_meta_a,
    )

    # Validation
    assert len(val_probs_rf) == Config.DEBUG_SIZE
    assert len(test_probs_rf) == Config.DEBUG_SIZE
    assert hasattr(
        model_rf, "predict_proba"
    ), "Model returned is not a valid classifier"
    print("Stream A execution complete.")

    # --------------------------------------------------------------------------
    # 7. Model Training: Stream B (Attention-Gated MLP)
    # --------------------------------------------------------------------------
    print("\n[Step 6] Running Stream B (Attention-Gated MLP)...")

    # Train
    model_mlp = train_mlp(
        train_text,
        train_subs,
        train_meta_b,
        train_y,
        val_text,
        val_subs,
        val_meta_b,
        val_y,
    )

    # Predict
    val_probs_mlp = predict_mlp(model_mlp, val_text, val_subs, val_meta_b)
    test_probs_mlp = predict_mlp(model_mlp, test_text, test_subs, test_meta_b)

    # Validation
    assert isinstance(model_mlp, torch.nn.Module), "Model is not a PyTorch Module"
    assert len(val_probs_mlp) == Config.DEBUG_SIZE
    assert len(test_probs_mlp) == Config.DEBUG_SIZE
    print("Stream B execution complete.")

    # --------------------------------------------------------------------------
    # 8. Ensembling and Submission
    # --------------------------------------------------------------------------
    print("\n[Step 7] Generating Submission...")

    # Simple weighted ensemble
    w_rf = Config.ENSEMBLE_WEIGHTS["rf"]
    w_mlp = Config.ENSEMBLE_WEIGHTS["mlp"]

    final_preds = (w_rf * test_probs_rf) + (w_mlp * test_probs_mlp)

    # Save
    ids = df_test[Config.ID_COL].values
    save_submission(ids, final_preds)

    # Validation
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    # Check file content format
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert list(df_sub.columns) == [
        Config.ID_COL,
        Config.TARGET_COL,
    ], "Submission columns incorrect"
    assert len(df_sub) == Config.DEBUG_SIZE, "Submission row count incorrect"

    print(f"\n=== Demo Pipeline Completed Successfully ===")
    print(f"Submission saved to: {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    run_demo_pipeline()
