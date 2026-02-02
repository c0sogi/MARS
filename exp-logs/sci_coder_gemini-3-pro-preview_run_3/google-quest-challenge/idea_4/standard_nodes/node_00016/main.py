import os
import sys
import numpy as np
import pandas as pd
import torch

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, compute_metric
from library import engine


def main():
    # --------------------------------------------------------------------------
    # 1. Setup & Initialization
    # --------------------------------------------------------------------------
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --------------------------------------------------------------------------
    # 2. Stream 1: MPNet Backbone
    # --------------------------------------------------------------------------
    print(f"\n=== Processing Stream 1: {Config.MODEL_1_NAME} ===")

    # Train the backbone (or load if already cached/trained)
    model_1 = engine.train_stream(
        model_name=Config.MODEL_1_NAME,
        save_path=Config.M1_MODEL_PATH,
        device=device,
        load_cached_data=True,
    )

    # Extract features for Train, Val, and Test
    m1_cache_paths = (
        Config.M1_TRAIN_FEATS_PATH,
        Config.M1_VAL_FEATS_PATH,
        Config.M1_TEST_FEATS_PATH,
    )
    m1_train, m1_val, m1_test = engine.process_stream_features(
        model=model_1,
        model_name=Config.MODEL_1_NAME,
        device=device,
        cache_paths=m1_cache_paths,
        load_cached_data=True,
    )

    # Cleanup memory
    del model_1
    torch.cuda.empty_cache()

    # --------------------------------------------------------------------------
    # 3. Feature Preparation
    # --------------------------------------------------------------------------
    print("\n=== Preparing Features and Loading Targets ===")

    # Load targets from the processed parquet files to ensure alignment with features.
    train_parquet = os.path.join(Config.WORKING_DIR, "train_processed.parquet")
    val_parquet = os.path.join(Config.WORKING_DIR, "val_processed.parquet")

    if os.path.exists(train_parquet):
        df_train = pd.read_parquet(train_parquet)
        df_val = pd.read_parquet(val_parquet)
    else:
        # Fallback to metadata CSVs if parquet not found
        df_train = pd.read_csv(Config.TRAIN_PATH)
        df_val = pd.read_csv(Config.VAL_PATH)

    y_train = df_train[Config.TARGET_COLS].values
    y_val = df_val[Config.TARGET_COLS].values

    # Single Stream Features (Cite solution_lesson_node_00015: Avoid Feature Concatenation when p > n)
    X_train = m1_train
    X_val = m1_val
    X_test = m1_test

    print(f"Final Feature Matrix Shape (Train): {X_train.shape}")

    # --------------------------------------------------------------------------
    # 6. Ensemble Training (Ridge Regression)
    # --------------------------------------------------------------------------
    print("\n=== Training Ridge Ensemble ===")
    ridge_model = engine.run_ensemble(
        X_train, y_train, X_val, y_val, Config.RIDGE_MODEL_PATH
    )

    # --------------------------------------------------------------------------
    # 7. Validation & Metrics
    # --------------------------------------------------------------------------
    print("\n=== Validation Assessment ===")
    val_preds = ridge_model.predict(X_val)
    val_preds = np.clip(val_preds, 0, 1)

    final_score = compute_metric(y_val, val_preds)
    print(f"Final Validation Metric: {final_score}")

    # --------------------------------------------------------------------------
    # 8. Failure Analysis
    # --------------------------------------------------------------------------
    print("\n=== Failure Analysis ===")
    # Calculate Mean Absolute Error per sample
    errors = np.mean(np.abs(y_val - val_preds), axis=1)

    print(f"Mean Absolute Error: {np.mean(errors):.4f}")
    print(f"Max Error: {np.max(errors):.4f}")

    # --------------------------------------------------------------------------
    # 9. Submission Generation
    # --------------------------------------------------------------------------
    threshold = 0.39777746135407066

    if final_score > threshold:
        print(
            f"\nValidation score ({final_score}) exceeds threshold ({threshold}). Generating submission..."
        )
        engine.generate_submission(ridge_model, X_test, Config.SUBMISSION_FILE_PATH)
    else:
        print(
            f"\nValidation score ({final_score}) does not meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
