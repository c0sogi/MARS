import os
import json
import pandas as pd
import numpy as np
from typing import Dict

from library.config import Config
from library.utils import seed_everything
from library.data_loader import load_metadata, load_sample_submission
from library.feature_engineering import FeatureEngineer
from library.model import DualStreamModel


def generate_predictions(
    debug_mode: bool = False, load_cached_data: bool = True
) -> None:
    """
    Generates predictions for the test set using the trained Dual-Stream GBDT models.

    1. Loads test metadata and optimized thresholds.
    2. Generates features for Stream A (Interaction) and Stream B (Impact).
    3. Predicts probabilities using the respective trained models.
    4. Applies optimized thresholds to generate binary labels.
    5. Formats and saves the submission file.

    Args:
        debug_mode (bool): If True, processes a subset of data for debugging.
        load_cached_data (bool): If True, attempts to load features from cache.
    """
    seed_everything(Config.SEED)

    print("Initializing Inference Pipeline...")

    # 1. Load Test Metadata
    df_test_meta = load_metadata("test")

    if debug_mode:
        print("DEBUG MODE: Sampling subset of test games...")
        sample_games = df_test_meta["game_play"].unique()[:2]
        df_test_meta = df_test_meta[df_test_meta["game_play"].isin(sample_games)].copy()
        print(f"Test Meta Shape: {df_test_meta.shape}")

    # 2. Load Thresholds
    thresholds_path = os.path.join(Config.WORKING_DIR, "thresholds.json")
    if os.path.exists(thresholds_path):
        with open(thresholds_path, "r") as f:
            thresholds = json.load(f)
        print(f"Loaded thresholds: {thresholds}")
    else:
        print(
            f"Warning: Thresholds file not found at {thresholds_path}. Defaulting to 0.5."
        )
        thresholds = {
            Config.STREAM_A["name"]: Config.DEFAULT_THRESHOLD,
            Config.STREAM_B["name"]: Config.DEFAULT_THRESHOLD,
        }

    # Initialize Components
    fe = FeatureEngineer()
    model_wrapper = DualStreamModel()

    # Dictionary to store predictions: contact_id -> {prob, stream}
    prediction_map: Dict[str, Dict[str, Any]] = {}

    # =========================================================================
    # Stream A: Interaction Model (Player vs Player)
    # =========================================================================
    print("\n" + "=" * 40)
    print("Inference Stream A: Interaction Model")
    print("=" * 40)

    X_test_a, _, ids_test_a = fe.create_features(
        metadata_df=df_test_meta,
        stream_config=Config.STREAM_A,
        dataset_type="test",
        load_cached_data=load_cached_data,
    )

    if not X_test_a.empty:
        probs_a = model_wrapper.predict_stream(X_test_a, Config.STREAM_A)

        for cid, prob in zip(ids_test_a, probs_a):
            prediction_map[cid] = {"prob": prob, "stream": Config.STREAM_A["name"]}
        print(f"[Stream A] Generated {len(probs_a)} predictions.")
    else:
        print("[Stream A] No relevant test samples found.")

    # =========================================================================
    # Stream B: Impact Model (Player vs Ground)
    # =========================================================================
    print("\n" + "=" * 40)
    print("Inference Stream B: Impact Model")
    print("=" * 40)

    X_test_b, _, ids_test_b = fe.create_features(
        metadata_df=df_test_meta,
        stream_config=Config.STREAM_B,
        dataset_type="test",
        load_cached_data=load_cached_data,
    )

    if not X_test_b.empty:
        probs_b = model_wrapper.predict_stream(X_test_b, Config.STREAM_B)

        for cid, prob in zip(ids_test_b, probs_b):
            prediction_map[cid] = {"prob": prob, "stream": Config.STREAM_B["name"]}
        print(f"[Stream B] Generated {len(probs_b)} predictions.")
    else:
        print("[Stream B] No relevant test samples found.")

    # =========================================================================
    # Submission Assembly
    # =========================================================================
    print("\n" + "=" * 40)
    print("Assembling Submission")
    print("=" * 40)

    # Load sample submission to ensure correct order and completeness
    df_submission = load_sample_submission()

    # If in debug mode, filter submission to match the sampled metadata
    if debug_mode:
        df_submission = df_submission[
            df_submission["contact_id"].isin(df_test_meta["contact_id"])
        ].copy()

    final_contacts = []
    missing_count = 0

    for cid in df_submission["contact_id"]:
        if cid in prediction_map:
            pred_info = prediction_map[cid]
            prob = pred_info["prob"]
            stream_name = pred_info["stream"]
            thresh = thresholds.get(stream_name, 0.5)

            # Apply threshold
            label = 1 if prob >= thresh else 0
            final_contacts.append(label)
        else:
            # Fallback for missing predictions (should not happen in full run)
            final_contacts.append(0)
            missing_count += 1

    df_submission["contact"] = final_contacts

    if missing_count > 0:
        print(f"Warning: {missing_count} predictions were missing and defaulted to 0.")

    # Save Submission
    save_path = Config.SUBMISSION_PATH
    df_submission.to_csv(save_path, index=False)

    print(f"Submission saved to {save_path}")
    print(f"Total rows: {len(df_submission)}")
    print("Inference pipeline completed successfully.")
