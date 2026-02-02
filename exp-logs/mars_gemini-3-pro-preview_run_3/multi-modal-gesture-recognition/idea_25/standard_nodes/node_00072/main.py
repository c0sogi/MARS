import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import scipy.stats

# Import library modules
import library.config as config
import library.trainer as trainer
import library.inference as inference
import library.data_utils as data_utils
from library.model import NRGSNet
from library.config import (
    VAL_METADATA_PATH,
    INPUT_DIR,
    NUM_CLASSES,
    SEED,
    MIN_GESTURE_DURATION,
    WORKING_DIR,
)


# ==========================================
# 1. Setup & Configuration
# ==========================================
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)

# Monkeypatch config for fast baseline execution
# Reducing epochs to ensure completion within 2 hours while allowing convergence
config.EPOCHS = 35
print(f"Adjusted EPOCHS to {config.EPOCHS} for fast baseline.")


# ==========================================
# 2. Helper Functions
# ==========================================
def calculate_levenshtein(seq1, seq2):
    """
    Calculates the Levenshtein distance between two sequences.
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
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1, matrix[x - 1, y - 1], matrix[x, y - 1] + 1
                )
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1, matrix[x - 1, y - 1] + 1, matrix[x, y - 1] + 1
                )
    return matrix[size_x - 1, size_y - 1]


def run_validation_inference(model_path):
    """
    Runs inference on validation set and calculates the competition metric.
    Returns:
        float: The final metric score.
        pd.DataFrame: DataFrame containing per-sample error stats for failure analysis.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Validation Inference Device: {device}")

    # Load Model
    model = NRGSNet().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Load Validation Metadata
    df_val = pd.read_csv(VAL_METADATA_PATH)

    total_distance = 0
    total_gestures = 0

    analysis_data = []

    print(f"Evaluating on {len(df_val)} validation sequences...")

    for _, row in df_val.iterrows():
        sample_id = row["sample_id"]
        data_path = os.path.join(INPUT_DIR, row["data_path"])
        audio_path = os.path.join(INPUT_DIR, row["audio_path"])

        # Parse Ground Truth
        gt_labels = []
        if pd.notna(row["labels"]):
            try:
                label_list = json.loads(row["labels"])
                # Sort by begin frame just in case
                label_list.sort(key=lambda x: x["begin"])
                gt_labels = [int(l["id"]) for l in label_list]
            except:
                gt_labels = []

        # --- Data Loading & Inference ---
        # 1. Load Skeleton
        skeleton = data_utils.load_robust_mat(data_path, load_cached_data=True)

        predicted_gestures = []

        if skeleton is not None and skeleton.shape[0] > 0:
            num_frames = skeleton.shape[0]

            # 2. Load Audio
            audio = data_utils.extract_audio_mfcc(
                audio_path, num_frames, load_cached_data=True
            )

            # 3. Compute Kinematics
            kinematics = data_utils.compute_kinematics(skeleton)
            T, J, D = kinematics.shape
            kinematics_flat = kinematics.reshape(T, J * D)

            # 4. Fusion
            features = np.concatenate([kinematics_flat, audio], axis=-1)

            # 5. Predict
            avg_probs = inference.predict_sliding_window(model, features, device)
            frame_preds = np.argmax(avg_probs, axis=1)
            predicted_gestures = inference.post_process_predictions(frame_preds)

        # --- Metric Calculation ---
        dist = calculate_levenshtein(predicted_gestures, gt_labels)

        total_distance += dist
        total_gestures += len(gt_labels)

        # Collect stats for failure analysis
        analysis_data.append(
            {
                "sample_id": sample_id,
                "levenshtein_dist": dist,
                "num_gt_gestures": len(gt_labels),
                "num_pred_gestures": len(predicted_gestures),
                "seq_duration": skeleton.shape[0] if skeleton is not None else 0,
                "error_rate": (
                    dist / len(gt_labels)
                    if len(gt_labels) > 0
                    else (1.0 if dist > 0 else 0.0)
                ),
            }
        )

    # Avoid division by zero
    final_metric = total_distance / total_gestures if total_gestures > 0 else 1.0

    return final_metric, pd.DataFrame(analysis_data)


# ==========================================
# 3. Main Execution
# ==========================================
if __name__ == "__main__":
    print("=== Starting Runfile Execution ===")
    start_total = time.time()

    # --- Step 1: Training ---
    print("\n--- Step 1: Training Model ---")
    # We use all data (limit_data=None) but reduced epochs via config patch
    best_model_path = trainer.train_model(limit_data=None, load_cached_data=True)

    # --- Step 2: Validation & Metric ---
    print("\n--- Step 2: Validation & Metric Calculation ---")
    val_metric, analysis_df = run_validation_inference(best_model_path)

    print(f"Final Validation Metric: {val_metric:.10f}")

    # --- Step 3: Failure Analysis ---
    print("\n--- Step 3: Failure Analysis ---")
    if not analysis_df.empty:
        # Correlation between error magnitude and sequence properties
        corr_dur, _ = scipy.stats.pearsonr(
            analysis_df["levenshtein_dist"], analysis_df["seq_duration"]
        )
        corr_cnt, _ = scipy.stats.pearsonr(
            analysis_df["levenshtein_dist"], analysis_df["num_gt_gestures"]
        )

        print(f"Correlation (Error vs Duration): {corr_dur:.4f}")
        print(f"Correlation (Error vs Num Gestures): {corr_cnt:.4f}")

        # Check for empty predictions
        empty_preds = analysis_df[analysis_df["num_pred_gestures"] == 0]
        print(f"Samples with 0 predicted gestures: {len(empty_preds)}")

        # High error samples
        high_error = analysis_df.sort_values("levenshtein_dist", ascending=False).head(
            5
        )
        print("Top 5 High Error Samples:")
        print(
            high_error[
                [
                    "sample_id",
                    "levenshtein_dist",
                    "num_gt_gestures",
                    "num_pred_gestures",
                ]
            ].to_string(index=False)
        )

    # --- Step 4: Submission ---
    print("\n--- Step 4: Submission Generation ---")
    THRESHOLD = 0.2251

    if val_metric < THRESHOLD:
        print(
            f"Validation metric {val_metric:.4f} is below threshold {THRESHOLD}. Generating submission..."
        )
        submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
        inference.generate_submission(
            model_path=best_model_path,
            output_file=submission_path,
            load_cached_data=True,
        )
    else:
        print(
            f"Validation metric {val_metric:.4f} did not meet threshold {THRESHOLD}. Skipping submission."
        )

    end_total = time.time()
    print(f"\nTotal Execution Time: {(end_total - start_total)/60:.2f} minutes")
