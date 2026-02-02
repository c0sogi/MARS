import os
import sys
import pandas as pd
import numpy as np
import torch
import scipy.stats

# Ensure the current directory is in the path to import library modules
sys.path.append(os.getcwd())

import library.config as config
import library.utils as utils
import library.data_loader as data_loader
import library.model as model_lib
import library.trainer as trainer_lib
import library.inference as inference_lib


def levenshtein_distance(s1, s2):
    """
    Computes the Levenshtein distance between two sequences.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def main():
    # 1. Setup and Configuration
    utils.set_seed(config.SEED)
    device = utils.get_device()
    print(f"Using device: {device}")

    # Configure for a robust baseline run
    # Using full dataset (limit=None) as the dataset is small enough for the A100 (approx 1700 samples)
    # Increasing epochs to ensure convergence within the time limit
    config.TRAIN_CONFIG["batch_size"] = 16
    config.TRAIN_CONFIG["num_epochs"] = 35
    config.TRAIN_CONFIG["patience"] = 10

    # 2. Data Loading
    print("Loading Data...")
    # limit=None ensures we use the full training and validation sets
    train_loader, val_loader, test_loader = data_loader.get_dataloaders(
        limit=None,
        batch_size=config.TRAIN_CONFIG["batch_size"],
        num_workers=config.TRAIN_CONFIG["num_workers"],
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = model_lib.BMGCN()
    trainer = trainer_lib.Trainer(model, device)

    checkpoint_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    # 4. Training
    print("Starting Training...")
    trainer.fit(
        train_loader,
        val_loader,
        epochs=config.TRAIN_CONFIG["num_epochs"],
        patience=config.TRAIN_CONFIG["patience"],
        checkpoint_path=checkpoint_path,
    )

    # 5. Validation and Metric Calculation
    print("Evaluating Validation Set...")
    # Load the best model weights
    utils.load_checkpoint(checkpoint_path, model, device=device)

    # Generate predictions on validation set
    # trainer.predict returns a list of (sample_id, decoded_sequence)
    val_results = trainer.predict(val_loader)

    # Load Ground Truth from metadata
    val_df = pd.read_csv(config.VAL_METADATA_PATH)
    # Convert space-separated string labels to list of integers
    val_df["labels"] = val_df["labels"].apply(
        lambda x: (
            [int(i) for i in str(x).split()]
            if pd.notna(x) and str(x).strip() != ""
            else []
        )
    )
    gt_map = dict(zip(val_df["sample_id"], val_df["labels"]))

    total_edit_distance = 0
    total_gt_gestures = 0
    analysis_records = []

    for sample_id, pred_seq in val_results:
        if sample_id not in gt_map:
            continue

        gt_seq = gt_map[sample_id]
        dist = levenshtein_distance(pred_seq, gt_seq)

        total_edit_distance += dist
        total_gt_gestures += len(gt_seq)

        # Collect data for failure analysis
        row = val_df[val_df["sample_id"] == sample_id].iloc[0]
        analysis_records.append(
            {
                "sample_id": sample_id,
                "error": dist,
                "num_frames": row["num_frames"],
                "num_gt": len(gt_seq),
            }
        )

    # Compute final metric: Total Edit Distance / Total Number of Gestures
    final_metric = (
        total_edit_distance / total_gt_gestures if total_gt_gestures > 0 else 1.0
    )
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("Performing Failure Analysis...")
    if analysis_records:
        df_analysis = pd.DataFrame(analysis_records)

        # Correlation: Error vs Sequence Length (Frames)
        if len(df_analysis) > 1:
            corr_frames, _ = scipy.stats.pearsonr(
                df_analysis["error"], df_analysis["num_frames"]
            )
            print(f"Correlation (Error vs Input Features - NumFrames): {corr_frames}")

            # Correlation: Error vs Number of Gestures
            corr_gestures, _ = scipy.stats.pearsonr(
                df_analysis["error"], df_analysis["num_gt"]
            )
            print(
                f"Correlation (Error vs Input Features - NumGestures): {corr_gestures}"
            )
        else:
            print("Not enough samples for correlation analysis.")

    # 7. Submission Generation
    # Threshold check as per requirements
    THRESHOLD = 0.06789606035205364

    if final_metric < THRESHOLD:
        print(
            f"Validation metric {final_metric} is below threshold {THRESHOLD}. Generating submission..."
        )
        inference_engine = inference_lib.InferenceEngine(checkpoint_path, device)
        submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
        inference_engine.run(
            test_metadata_path=config.TEST_METADATA_PATH,
            output_path=submission_path,
            batch_size=config.TRAIN_CONFIG["batch_size"],
            num_workers=config.TRAIN_CONFIG["num_workers"],
        )
    else:
        print(
            f"Validation metric {final_metric} is not lower than {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
