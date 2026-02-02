import sys
import os
import pandas as pd
import numpy as np
import torch

# Ensure library is in path
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, setup_logger
from library.trainer import train_transformer
from library.inference import CascadeInference
from library.hfbb import HFBB


def main():
    # 1. Setup
    logger = setup_logger("RunFile")
    set_seed(Config.SEED)

    # 2. Configure for Fast Baseline
    # We override default Config values to ensure the process completes within the time limit.
    # 5 Epochs is sufficient for a baseline on this large dataset.
    # 3 Folds reduces the overhead of the Jackknifing process for residual generation.
    Config.EPOCHS = 5
    Config.N_FOLDS = 3

    logger.info("Starting Pipeline Execution...")
    logger.info(
        f"Configuration: Epochs={Config.EPOCHS}, Folds={Config.N_FOLDS}, Device={Config.DEVICE}"
    )

    # 3. Train Transformer (Tier 2)
    # This handles BPE training, Curriculum Data Generation (Residuals + Anchors), and Transformer Training.
    logger.info(">>> Stage 1: Training Transformer (Tier 2)")
    try:
        train_transformer(load_cached_data=True)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        # We continue, as a model might already exist or we might want to test inference anyway
        # though usually this is fatal.
        raise e

    # 4. Ensure HFBB (Tier 1) is Ready
    # The inference engine needs the HFBB model fitted on the FULL training set.
    # While DataFactory fits HFBB on folds, we need the global map for final inference.
    logger.info(">>> Stage 2: Preparing HFBB (Tier 1)")
    if os.path.exists(Config.TRAIN_META):
        logger.info("Loading full training metadata for HFBB...")
        df_train_full = pd.read_csv(Config.TRAIN_META)
        hfbb = HFBB()
        # fit will load from cache if available, or build and save if not
        hfbb.fit(df_train_full, load_cached_data=True)
        del df_train_full  # Free memory
    else:
        logger.warning(
            "Training metadata not found. HFBB initialization relies solely on cache."
        )

    # 5. Initialize Inference Engine
    # This combines Tier 1 (HFBB) and Tier 2 (Transformer)
    logger.info(">>> Stage 3: Initializing Cascade Inference")
    inference_engine = CascadeInference()

    # 6. Validation
    logger.info(">>> Stage 4: Validation")
    val_path = Config.VAL_META
    if not os.path.exists(val_path):
        logger.error(f"Validation file not found at {val_path}")
        return

    logger.info(f"Loading validation data from {val_path}...")
    df_val = pd.read_csv(val_path)

    # Run Inference
    # context_source_path is required to reconstruct prev/next context for the Transformer
    logger.info("Running inference on validation set...")
    preds = inference_engine.predict_batch(df_val, context_source_path=val_path)

    # Compute Metric
    logger.info("Computing validation metrics...")
    df_val["pred"] = preds.fillna("")
    df_val["after"] = df_val["after"].fillna("").astype(str)
    df_val["pred"] = df_val["pred"].astype(str)

    # Exact match accuracy
    correct_mask = df_val["after"] == df_val["pred"]
    accuracy = correct_mask.mean()

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {accuracy}")

    # 7. Failure Analysis
    logger.info(">>> Stage 5: Failure Analysis")
    df_val["is_error"] = ~correct_mask

    # A. Correlation with Input Length
    df_val["len_before"] = df_val["before"].astype(str).str.len()
    corr_len = df_val["len_before"].corr(df_val["is_error"].astype(int))
    print(f"Correlation (Error vs Input Length): {corr_len:.10f}")

    # B. Error Rate by Class
    if "class" in df_val.columns:
        print("Error Rate by Class:")
        class_stats = df_val.groupby("class")["is_error"].agg(["mean", "count"])
        class_stats = class_stats.sort_values("mean", ascending=False)
        print(class_stats.head(10))

        # Identify worst performing class with significant volume
        significant_classes = class_stats[class_stats["count"] > 100]
        if not significant_classes.empty:
            worst_class = significant_classes.index[0]
            logger.info(
                f"Worst performing significant class: {worst_class} (Error Rate: {significant_classes.iloc[0]['mean']:.4f})"
            )

    # 8. Submission
    logger.info(">>> Stage 6: Submission Decision")
    THRESHOLD = 0.9784022349361615

    if accuracy > THRESHOLD:
        logger.info(
            f"Validation Accuracy ({accuracy:.6f}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        inference_engine.generate_submission()
    else:
        logger.info(
            f"Validation Accuracy ({accuracy:.6f}) does not exceed threshold ({THRESHOLD}). Submission skipped."
        )

    logger.info("Pipeline execution complete.")


if __name__ == "__main__":
    main()
