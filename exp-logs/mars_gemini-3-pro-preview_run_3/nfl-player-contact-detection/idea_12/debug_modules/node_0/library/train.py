import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any

from library.config import Config
from library.utils import seed_everything, optimize_threshold, calc_mcc
from library.data_loader import load_metadata
from library.feature_engineering import FeatureEngineer
from library.model import DualStreamModel


def run_training_pipeline(
    debug_mode: bool = False, load_cached_data: bool = True
) -> None:
    """
    Orchestrates the training pipeline for the Dual-Stream GBDT solution.

    1. Loads metadata.
    2. Generates features for Stream A (Interaction) and Stream B (Impact).
    3. Trains XGBoost models for each stream.
    4. Optimizes decision thresholds on the validation set.
    5. Calculates global validation metrics.
    6. Saves thresholds for inference.

    Args:
        debug_mode (bool): If True, runs on a small subset of data for debugging.
        load_cached_data (bool): If True, attempts to load features from Parquet cache.
    """
    seed_everything(Config.SEED)

    print("Initializing Training Pipeline...")

    # 1. Load Metadata
    df_train_meta = load_metadata("train")
    df_val_meta = load_metadata("validation")

    # Debugging: Sample data if requested
    if debug_mode:
        print("DEBUG MODE: Sampling subset of games...")
        sample_games_train = df_train_meta["game_play"].unique()[:5]
        sample_games_val = df_val_meta["game_play"].unique()[:2]

        df_train_meta = df_train_meta[
            df_train_meta["game_play"].isin(sample_games_train)
        ].copy()
        df_val_meta = df_val_meta[
            df_val_meta["game_play"].isin(sample_games_val)
        ].copy()

        print(f"Train Meta Shape: {df_train_meta.shape}")
        print(f"Val Meta Shape: {df_val_meta.shape}")

    # Initialize Components
    fe = FeatureEngineer()
    model_wrapper = DualStreamModel()

    # Store results for global evaluation
    # Map contact_id -> {prob, stream}
    val_predictions_map = {}

    # Store optimized thresholds
    thresholds = {}

    # =========================================================================
    # Stream A: Interaction Model (Player vs Player)
    # =========================================================================
    print("\n" + "=" * 40)
    print("Processing Stream A: Interaction Model")
    print("=" * 40)

    # Generate Features
    X_train_a, y_train_a, ids_train_a = fe.create_features(
        metadata_df=df_train_meta,
        stream_config=Config.STREAM_A,
        dataset_type="train",
        load_cached_data=load_cached_data,
    )

    X_val_a, y_val_a, ids_val_a = fe.create_features(
        metadata_df=df_val_meta,
        stream_config=Config.STREAM_A,
        dataset_type="validation",
        load_cached_data=load_cached_data,
    )

    if not X_train_a.empty:
        # Train Model
        model_wrapper.train_stream(
            X_train_a, y_train_a, X_val_a, y_val_a, Config.STREAM_A
        )

        # Predict on Validation
        probs_val_a = model_wrapper.predict_stream(X_val_a, Config.STREAM_A)

        # Optimize Threshold
        best_thresh_a, best_score_a = optimize_threshold(y_val_a, probs_val_a)
        thresholds[Config.STREAM_A["name"]] = best_thresh_a

        print(f"[Stream A] Optimal Threshold: {best_thresh_a:.4f}")
        print(f"[Stream A] Best MCC: {best_score_a}")

        # Store predictions for global eval
        for cid, prob in zip(ids_val_a, probs_val_a):
            val_predictions_map[cid] = {"prob": prob, "stream": Config.STREAM_A["name"]}
    else:
        print("[Stream A] No data found (likely due to debug sampling). Skipping.")
        thresholds[Config.STREAM_A["name"]] = 0.5

    # =========================================================================
    # Stream B: Impact Model (Player vs Ground)
    # =========================================================================
    print("\n" + "=" * 40)
    print("Processing Stream B: Impact Model")
    print("=" * 40)

    # Generate Features
    X_train_b, y_train_b, ids_train_b = fe.create_features(
        metadata_df=df_train_meta,
        stream_config=Config.STREAM_B,
        dataset_type="train",
        load_cached_data=load_cached_data,
    )

    X_val_b, y_val_b, ids_val_b = fe.create_features(
        metadata_df=df_val_meta,
        stream_config=Config.STREAM_B,
        dataset_type="validation",
        load_cached_data=load_cached_data,
    )

    if not X_train_b.empty:
        # Train Model
        model_wrapper.train_stream(
            X_train_b, y_train_b, X_val_b, y_val_b, Config.STREAM_B
        )

        # Predict on Validation
        probs_val_b = model_wrapper.predict_stream(X_val_b, Config.STREAM_B)

        # Optimize Threshold
        best_thresh_b, best_score_b = optimize_threshold(y_val_b, probs_val_b)
        thresholds[Config.STREAM_B["name"]] = best_thresh_b

        print(f"[Stream B] Optimal Threshold: {best_thresh_b:.4f}")
        print(f"[Stream B] Best MCC: {best_score_b}")

        # Store predictions for global eval
        for cid, prob in zip(ids_val_b, probs_val_b):
            val_predictions_map[cid] = {"prob": prob, "stream": Config.STREAM_B["name"]}
    else:
        print("[Stream B] No data found. Skipping.")
        thresholds[Config.STREAM_B["name"]] = 0.5

    # =========================================================================
    # Global Evaluation
    # =========================================================================
    print("\n" + "=" * 40)
    print("Global Validation Evaluation")
    print("=" * 40)

    # Reconstruct predictions aligned with original validation metadata
    y_true_global = []
    y_pred_global = []

    # We iterate over the original validation metadata to ensure we cover all cases
    # and maintain the correct order/population
    missing_preds = 0

    for _, row in df_val_meta.iterrows():
        cid = row["contact_id"]
        target = int(row["contact"])
        y_true_global.append(target)

        if cid in val_predictions_map:
            pred_info = val_predictions_map[cid]
            prob = pred_info["prob"]
            stream = pred_info["stream"]
            thresh = thresholds[stream]

            # Apply specific threshold
            pred_bin = 1 if prob >= thresh else 0
            y_pred_global.append(pred_bin)
        else:
            # Should not happen if features are generated correctly for all metadata rows
            # Default to 0 if missing
            y_pred_global.append(0)
            missing_preds += 1

    if missing_preds > 0:
        print(f"Warning: {missing_preds} validation samples were missing predictions.")

    global_mcc = calc_mcc(np.array(y_true_global), np.array(y_pred_global))
    print(f"Global Validation MCC: {global_mcc}")

    # =========================================================================
    # Save Artifacts
    # =========================================================================
    thresholds_path = os.path.join(Config.WORKING_DIR, "thresholds.json")
    with open(thresholds_path, "w") as f:
        json.dump(thresholds, f, indent=4)

    print(f"\nThresholds saved to {thresholds_path}")
    print("Training pipeline completed successfully.")
