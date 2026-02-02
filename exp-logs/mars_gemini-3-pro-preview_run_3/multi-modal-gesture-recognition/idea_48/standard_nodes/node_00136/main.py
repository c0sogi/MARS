import sys
import os
import json
import pandas as pd
import numpy as np
import torch
from scipy.stats import pearsonr

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, levenshtein_distance
from library.data_loader import get_loaders, process_sequence
from library.trainer import Trainer
from library.inference import InferenceEngine, generate_submission


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Adjust Config for a fast baseline run
    Config.EPOCHS = 15
    Config.NUM_WORKERS = 2

    # Set seeds for reproducibility
    set_seed(Config.SEED)

    print("Configuration:")
    print(f"  Epochs: {Config.EPOCHS}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  Device: {Config.DEVICE}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("\nLoading Data...")
    # load_cached_data=True uses pre-processed .npz files if available
    loaders = get_loaders(load_cached_data=True)

    # ==========================================
    # 3. Training
    # ==========================================
    print("\nInitializing Trainer...")
    trainer = Trainer(loaders["train"], loaders["val"])

    print("Starting Training...")
    trainer.fit(epochs=Config.EPOCHS)

    # ==========================================
    # 4. Validation & Metric Calculation
    # ==========================================
    print("\nPerforming Final Validation on Full Sequences...")

    # Initialize Inference Engine (loads best_model.pth automatically)
    try:
        engine = InferenceEngine()
    except FileNotFoundError:
        print("Best model checkpoint not found. Training might have failed.")
        return

    # Load Validation Metadata
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    predictions = []
    ground_truths = []
    errors = []

    # Meta-features for failure analysis
    meta_stats = {"duration": [], "num_gestures": []}

    print(f"Evaluating on {len(val_df)} validation samples...")

    for _, row in val_df.iterrows():
        # 1. Get Ground Truth
        labels_json = json.loads(row["labels"])
        gt_ids = [l["id"] for l in labels_json]
        ground_truths.append(gt_ids)

        # 2. Get Features
        # process_sequence returns (features, labels_array)
        # We pass is_train=False to disable augmentations like rotation
        feat, _ = process_sequence(row, is_train=False)

        if feat is None:
            # Handle edge cases (e.g., video too short or missing)
            pred_ids = []
            duration = 0
        else:
            # 3. Predict
            pred_ids = engine.predict_sequence(feat)
            duration = feat.shape[0]

        predictions.append(pred_ids)

        # 4. Compute Levenshtein Distance for this sample
        dist = levenshtein_distance(pred_ids, gt_ids)
        errors.append(dist)

        # Collect stats
        meta_stats["duration"].append(duration)
        meta_stats["num_gestures"].append(row["num_gestures"])

    # Calculate Final Metric (Normalized Levenshtein Distance)
    total_gestures = sum(len(gt) for gt in ground_truths)
    total_distance = sum(errors)

    # Avoid division by zero
    if total_gestures > 0:
        final_metric = total_distance / total_gestures
    else:
        final_metric = 1.0

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # 5. Failure Analysis
    # ==========================================
    print("\n=== Failure Analysis ===")
    if len(errors) > 1:
        # Correlation: Error vs Sequence Duration
        corr_dur, _ = pearsonr(errors, meta_stats["duration"])
        print(f"Correlation (Error vs Sequence Length): {corr_dur:.4f}")

        # Correlation: Error vs Number of Gestures
        corr_num, _ = pearsonr(errors, meta_stats["num_gestures"])
        print(f"Correlation (Error vs Num Gestures): {corr_num:.4f}")

        if abs(corr_dur) > 0.3:
            print("Observation: Model performance is sensitive to sequence length.")
        if abs(corr_num) > 0.3:
            print("Observation: Model performance is sensitive to gesture density.")
    else:
        print("Not enough samples for correlation analysis.")

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    THRESHOLD = 0.16539050535987748

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric:.10f}) is below threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(load_cached_data=True)
    else:
        print(
            f"\nMetric ({final_metric:.10f}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
