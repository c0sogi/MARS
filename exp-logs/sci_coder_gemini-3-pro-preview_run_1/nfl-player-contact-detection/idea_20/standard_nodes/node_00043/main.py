import os
import gc
import pandas as pd
import numpy as np
from library.config import SEED, SUBMISSION_OUTPUT_PATH, SENTINEL_VALUE
from library.utils import seed_everything, setup_logger
from library.data_processing import DataProcessor
from library.gating_filters import GatingFilter
from library.feature_engineering import FeatureEngineer
from library.trainer import Trainer
from library.evaluation import Evaluator


def main():
    # 1. Initialization
    seed_everything(SEED)
    logger = setup_logger("runfile")
    logger.info("Starting RKS-MTE Orchestration...")

    # 2. Data Processing (Training)
    logger.info("--- Stage 1: Processing Training Data ---")
    dp = DataProcessor()
    # Limit training samples for fast baseline execution
    df_train = dp.load_and_merge_data(
        "train", sample_size=500000, load_cached_data=True
    )

    gf = GatingFilter()
    df_train_gated = gf.apply_gating(df_train, load_cached_data=True)

    # Clean up raw train data
    del df_train
    gc.collect()

    fe = FeatureEngineer()
    df_train_feats = fe.create_features(
        df_train_gated, split="train", load_cached_data=True
    )

    # Clean up gated train data
    del df_train_gated
    gc.collect()

    # 3. Data Processing (Validation)
    logger.info("--- Stage 2: Processing Validation Data ---")
    # Use a large enough sample for valid metrics, but limit for speed
    df_val = dp.load_and_merge_data("val", sample_size=200000, load_cached_data=True)

    # We apply gating to validation to mimic the pipeline efficiency
    # Non-survivors will be assigned probability 0 later
    df_val_gated = gf.apply_gating(df_val, load_cached_data=True)

    # Keep track of indices for reconstruction
    # We need to know which rows in df_val (original) survived
    # df_val_gated is a subset of df_val.
    # We'll use contact_id as the key if needed, or just index preservation if carefully managed.
    # The GatingFilter returns a new dataframe.
    # Strategy: We will evaluate metrics on the FULL df_val loaded.
    # We need to map predictions back.

    # Generate features for survivors
    df_val_feats = fe.create_features(df_val_gated, split="val", load_cached_data=True)

    # 4. Training
    logger.info("--- Stage 3: Training Curriculum ---")
    trainer = Trainer()
    # Note: We pass df_val_feats to the trainer for early stopping on hard examples
    ensemble = trainer.run_curriculum(
        df_train_feats, df_val_feats, load_cached_data=True
    )

    # Clean up training features to free memory
    del df_train_feats
    gc.collect()

    # 5. Validation & Threshold Optimization
    logger.info("--- Stage 4: Evaluation & Threshold Optimization ---")
    evaluator = Evaluator()

    # Predict on Validation Survivors
    # Identify feature columns
    feature_cols = [
        c
        for c in df_val_feats.columns
        if c
        not in [
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "contact",
            "datetime",
            "video_path_endzone",
            "video_path_sideline",
            "video_path_all29",
            "offset",
            "step_actual",
        ]
        and np.issubdtype(df_val_feats[c].dtype, np.number)
    ]

    val_survivor_probs = ensemble.predict_proba(df_val_feats[feature_cols])

    # Reconstruct Full Validation Predictions
    # Create a series for predictions indexed by contact_id
    # 1. Create mapping from contact_id to prob for survivors
    survivor_map = dict(zip(df_val_feats["contact_id"], val_survivor_probs))

    # 2. Iterate over original df_val to build full vectors
    # Ensure contact_id exists in df_val
    if "contact_id" not in df_val.columns:
        df_val["contact_id"] = (
            df_val["game_play"]
            + "_"
            + df_val["step"].astype(str)
            + "_"
            + df_val["nfl_player_id_1"].astype(str)
            + "_"
            + df_val["nfl_player_id_2"].astype(str)
        )

    y_true_full = df_val["contact"].values
    # Default probability is 0.0 (for those gated out)
    y_pred_full = df_val["contact_id"].map(survivor_map).fillna(0.0).values

    # Optimize Threshold
    best_threshold, best_mcc = evaluator.optimize_threshold(y_true_full, y_pred_full)

    print(f"Final Validation Metric: {best_mcc}")

    # 6. Failure Analysis
    logger.info("--- Stage 5: Failure Analysis ---")
    # Analyze errors on survivors (where the model actually ran)
    # Calculate absolute error
    y_true_surv = df_val_feats["contact"].values
    errors = np.abs(y_true_surv - val_survivor_probs)

    # Correlate with features
    logger.info("Calculating feature correlations with error...")
    correlations = {}
    for col in feature_cols:
        # Simple correlation
        try:
            corr = np.corrcoef(df_val_feats[col].fillna(0), errors)[0, 1]
            if not np.isnan(corr):
                correlations[col] = corr
        except Exception:
            pass

    # Sort and print top correlations
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    logger.info("Top 10 Features correlated with Error:")
    for name, val in sorted_corr[:10]:
        logger.info(f"{name}: {val:.4f}")

    # 7. Submission
    logger.info("--- Stage 6: Submission Generation ---")

    if best_mcc > 0.6865:
        logger.info("Validation metric passed threshold. Generating submission...")

        # Load Test Data
        # We load full test data.
        df_test = dp.load_and_merge_data("test", load_cached_data=True)

        # Feature Engineering for Test
        # Per idea description: "Generate the Full Spectral Feature Set for the Entire Test Set"
        # We skip gating to ensure we have predictions for every row in sample_submission
        # or we gate and fill 0s. Gating is safer for runtime, but "Entire Test Set" implies full features.
        # However, sample_submission has 463k rows. Generating features for all is feasible.
        # We will skip gating for test to maximize potential recall and strictly follow the "Idea" text.

        # Note: FeatureEngineer expects a dataframe compatible with what comes out of gating.
        # df_test from load_and_merge_data is compatible.
        df_test_feats = fe.create_features(df_test, split="test", load_cached_data=True)

        # Predict
        test_probs = ensemble.predict_proba(df_test_feats[feature_cols])

        # Create mapping
        test_pred_map = dict(zip(df_test_feats["contact_id"], test_probs))

        # Load sample submission to ensure correct order and rows
        sample_sub = pd.read_csv("./input/sample_submission.csv")

        # Map predictions
        # Fill missing (if any dropped during processing) with 0
        sample_sub["contact"] = sample_sub["contact_id"].map(test_pred_map).fillna(0.0)

        # Apply Threshold
        sample_sub["contact"] = (sample_sub["contact"] >= best_threshold).astype(int)

        # Save
        sample_sub.to_csv(SUBMISSION_OUTPUT_PATH, index=False)
        logger.info(f"Submission saved to {SUBMISSION_OUTPUT_PATH}")

    else:
        logger.info(
            f"Validation metric {best_mcc} did not meet criteria (0.6865). Skipping submission."
        )


if __name__ == "__main__":
    main()
