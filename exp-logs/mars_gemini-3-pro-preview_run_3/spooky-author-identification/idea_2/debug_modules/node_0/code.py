import os
import pandas as pd
import numpy as np
import torch
import shutil

# Import from the provided library
from library.config import Config
from library.utils import set_seed
from library.data_loader import load_raw_data, get_dataloaders, get_tfidf_vectors
from library.models_stat import train_statistical_branch
from library.models_dl import train_transformer, predict_transformer
from library.ensemble import optimize_ensemble_weights, generate_and_save_submission


def run_pipeline_demonstration():
    print("=== Starting Authorship Attribution Pipeline Demonstration ===")

    # 1. Override Configuration for Speed
    print("\n[Step 1] Configuring environment for rapid demonstration...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 60  # Small subset for speed
    Config.EPOCHS = 1  # Single epoch for demonstration
    Config.MAX_LEN = 32  # Short sequence length
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 8

    # Ensure cache directory is clean for this run to avoid shape mismatches with debug data
    if os.path.exists(Config.CACHE_DIR):
        print(f"Cleaning cache directory: {Config.CACHE_DIR}")
        shutil.rmtree(Config.CACHE_DIR)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    set_seed(Config.SEED)
    print("Configuration updated: DEBUG=True, EPOCHS=1, MAX_LEN=32")

    # 2. Data Loading
    print("\n[Step 2] Loading Data...")
    train_df, val_df, test_df = load_raw_data(
        debug=Config.DEBUG, sample_size=Config.DEBUG_SAMPLE_SIZE
    )

    # Verification
    assert len(train_df) == Config.DEBUG_SAMPLE_SIZE, "Train DF size mismatch"
    assert len(val_df) == Config.DEBUG_SAMPLE_SIZE, "Val DF size mismatch"
    assert len(test_df) == Config.DEBUG_SAMPLE_SIZE, "Test DF size mismatch"
    print(f"Data loaded successfully. Train shape: {train_df.shape}")

    # 3. Statistical Branch
    print("\n[Step 3] Running Statistical Branch...")
    # Force computation (load_cached_vectors=False) to ensure it runs on the debug subset
    stat_model, X_test_stat = train_statistical_branch(
        train_df, val_df, test_df, load_cached_vectors=False
    )

    # Generate validation predictions for ensemble optimization
    # We reload the vectors we just computed/cached to get X_val
    _, X_val_stat, _ = get_tfidf_vectors(
        train_df, val_df, test_df, load_cached_data=True
    )
    p_val_stat = stat_model.predict_proba(X_val_stat)
    p_test_stat = stat_model.predict_proba(X_test_stat)

    # Verification
    assert p_val_stat.shape == (
        len(val_df),
        3,
    ), "Statistical validation probs shape mismatch"
    assert p_test_stat.shape == (
        len(test_df),
        3,
    ), "Statistical test probs shape mismatch"
    print("Statistical branch completed successfully.")

    # 4. Deep Learning Branch
    print("\n[Step 4] Running Deep Learning Branch (Transformer)...")
    train_loader, val_loader, test_loader = get_dataloaders(
        train_df,
        val_df,
        test_df,
        batch_size=Config.TRAIN_BATCH_SIZE,
        val_batch_size=Config.VALID_BATCH_SIZE,
        max_len=Config.MAX_LEN,
    )

    # Train the model
    dl_model = train_transformer(
        train_loader, val_loader, epochs=Config.EPOCHS, save_path=Config.MODEL_SAVE_PATH
    )

    # Generate predictions
    print("Generating Transformer predictions...")
    p_val_dl = predict_transformer(dl_model, val_loader)
    p_test_dl = predict_transformer(dl_model, test_loader)

    # Verification
    assert p_val_dl.shape == (len(val_df), 3), "DL validation probs shape mismatch"
    assert p_test_dl.shape == (len(test_df), 3), "DL test probs shape mismatch"
    print("Deep Learning branch completed successfully.")

    # 5. Ensemble Optimization
    print("\n[Step 5] Optimizing Ensemble...")
    y_val = val_df["author"].map(Config.LABEL_MAP).values

    best_weight = optimize_ensemble_weights(p_val_stat, p_val_dl, y_val)

    # Verification
    assert 0.0 <= best_weight <= 1.0, "Optimal weight out of bounds"
    print(f"Ensemble optimization complete. Best Transformer Weight: {best_weight}")

    # 6. Submission Generation
    print("\n[Step 6] Generating Submission...")
    test_ids = test_df["id"].values
    generate_and_save_submission(
        test_ids,
        p_test_stat,
        p_test_dl,
        best_weight,
        output_path=Config.SUBMISSION_PATH,
    )

    # Final Verification
    if os.path.exists(Config.SUBMISSION_PATH):
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission file created at {Config.SUBMISSION_PATH}")
        print(f"Submission shape: {sub_df.shape}")

        expected_cols = ["id"] + Config.CLASS_NAMES
        assert (
            list(sub_df.columns) == expected_cols
        ), f"Submission columns mismatch. Expected {expected_cols}, got {list(sub_df.columns)}"
        assert len(sub_df) == len(test_df), "Submission row count mismatch"

        # Check if probabilities are valid
        probs = sub_df[Config.CLASS_NAMES].values
        assert (probs >= 0).all() and (
            probs <= 1
        ).all(), "Probabilities out of range [0, 1]"

        print("=== Demonstration Completed Successfully ===")
    else:
        raise FileNotFoundError("Submission file was not created.")


if __name__ == "__main__":
    run_pipeline_demonstration()
