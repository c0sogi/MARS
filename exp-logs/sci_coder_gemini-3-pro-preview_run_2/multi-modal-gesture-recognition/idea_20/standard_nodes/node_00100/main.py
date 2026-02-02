import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.utils import set_seed
from library.trainer import Trainer
from library.inference import Predictor


def levenshtein_distance(seq1, seq2):
    """
    Computes the Levenshtein distance between two sequences of integers.
    """
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = matrix[x - 1, y - 1]
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1, matrix[x - 1, y - 1] + 1, matrix[x, y - 1] + 1
                )
    return matrix[size_x - 1, size_y - 1]


def setup_fast_run():
    """
    Sets up a temporary environment for a fast baseline run by subsampling training data.
    """
    # Define temp directories
    temp_meta_dir = "./working/temp_metadata"
    temp_cache_dir = "./working/temp_cache"

    if os.path.exists(temp_meta_dir):
        shutil.rmtree(temp_meta_dir)
    os.makedirs(temp_meta_dir)

    if os.path.exists(temp_cache_dir):
        shutil.rmtree(temp_cache_dir)
    os.makedirs(temp_cache_dir)

    # Load original metadata
    train_csv = os.path.join(Config.METADATA_DIR, "train.csv")
    val_csv = os.path.join(Config.METADATA_DIR, "val.csv")
    test_csv = os.path.join(Config.METADATA_DIR, "test.csv")

    df_train = pd.read_csv(train_csv)

    # Subsample training data (e.g., 1000 samples)
    # This limits the maximum number of training samples as requested
    subset_size = min(len(df_train), 1000)
    df_train_sub = df_train.sample(n=subset_size, random_state=Config.SEED).reset_index(
        drop=True
    )

    # Save modified metadata
    df_train_sub.to_csv(os.path.join(temp_meta_dir, "train.csv"), index=False)
    shutil.copy(val_csv, os.path.join(temp_meta_dir, "val.csv"))
    shutil.copy(test_csv, os.path.join(temp_meta_dir, "test.csv"))

    print(
        f"Fast run setup: Training data subsampled from {len(df_train)} to {len(df_train_sub)} samples."
    )

    # Override Config
    Config.METADATA_DIR = temp_meta_dir
    Config.CACHE_DIR = temp_cache_dir
    Config.NUM_EPOCHS = 8  # Reduced epochs for speed
    Config.BATCH_SIZE = 8


def main():
    # Set seed for reproducibility
    set_seed(Config.SEED)

    # Setup for fast execution
    setup_fast_run()

    # -------------------------------------------------------------------------
    # 1. Training
    # -------------------------------------------------------------------------
    print("\nStarting Training...")
    trainer = Trainer()
    trainer.fit()

    # -------------------------------------------------------------------------
    # 2. Validation & Metric Calculation
    # -------------------------------------------------------------------------
    print("\nStarting Validation Evaluation...")

    # Load best model
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")
    predictor = Predictor(checkpoint_path=best_model_path)

    # Run inference on validation set
    val_ids, val_preds = predictor.predict_dataset("val")

    # Load Ground Truth
    val_meta_path = os.path.join(Config.METADATA_DIR, "val.csv")
    df_val = pd.read_csv(val_meta_path)

    # Parse labels (string "1 2 3" -> list [1, 2, 3])
    df_val["labels"] = df_val["labels"].apply(
        lambda x: (
            [int(i) for i in str(x).split()]
            if pd.notna(x) and str(x).strip() != ""
            else []
        )
    )

    # Create a map for easy lookup
    pred_map = {sid: pred for sid, pred in zip(val_ids, val_preds)}

    total_distance = 0.0
    total_gestures = 0

    # Data for failure analysis
    analysis_data = []

    for _, row in df_val.iterrows():
        sid = row["sample_id"]
        ground_truth = row["labels"]
        prediction = pred_map.get(sid, [])

        # Compute Levenshtein Distance
        dist = levenshtein_distance(ground_truth, prediction)

        total_distance += dist
        total_gestures += len(ground_truth)

        analysis_data.append(
            {
                "sample_id": sid,
                "error": dist,
                "num_frames": row["num_frames"],
                "num_gestures": len(ground_truth),
            }
        )

    # Compute Final Metric
    # Avoid division by zero
    if total_gestures > 0:
        final_metric = total_distance / total_gestures
    else:
        final_metric = 0.0

    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 3. Failure Analysis
    # -------------------------------------------------------------------------
    print("\nPerforming Failure Analysis...")
    df_analysis = pd.DataFrame(analysis_data)

    if not df_analysis.empty:
        # Correlation between Error and Sequence Length (NumFrames)
        corr_frames = df_analysis["error"].corr(df_analysis["num_frames"])

        # Correlation between Error and Complexity (NumGestures)
        corr_gestures = df_analysis["error"].corr(df_analysis["num_gestures"])

        print(f"Correlation (Error vs NumFrames): {corr_frames}")
        print(f"Correlation (Error vs NumGestures): {corr_gestures}")
    else:
        print("No validation data available for analysis.")

    # -------------------------------------------------------------------------
    # 4. Submission
    # -------------------------------------------------------------------------
    THRESHOLD = 0.06789606035205364

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is below threshold ({THRESHOLD}). Generating submission..."
        )
        # Generate predictions for test set
        # The generate_submission method saves to Config.SUBMISSION_DIR/submission.csv
        predictor.generate_submission("submission.csv")
    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
