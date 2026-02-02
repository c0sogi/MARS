import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import json

# Import from library
from library.config import Config, seed_everything
from library.trainer import Trainer
from library.inference import InferenceEngine
from library.data_utils import get_feature_vector


# ==========================================
# 1. Configuration & Setup
# ==========================================
def setup_config():
    """Override Config for Fast Baseline Execution"""
    # Set a fixed seed
    seed_everything(42)

    # Limit epochs for speed (dataset is small, 15 epochs is usually enough for convergence)
    Config.NUM_EPOCHS = 15
    Config.BATCH_SIZE = 32

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)


# ==========================================
# 2. Metric Implementation (Levenshtein)
# ==========================================
def levenshtein_distance(seq1, seq2):
    """
    Calculates Levenshtein distance between two sequences.
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


def decode_sequence(pred_labels_frame_wise):
    """
    Decodes frame-wise labels into a list of gesture IDs using RLE.
    Removes background class (0).
    """
    decoded = []
    last_label = None

    for label in pred_labels_frame_wise:
        label = int(label)
        if label == Config.BACKGROUND_CLASS_ID:
            last_label = None
            continue

        if label != last_label:
            decoded.append(label)
            last_label = label

    return decoded


# ==========================================
# 3. Validation & Analysis
# ==========================================
def evaluate_and_analyze(trainer):
    """
    Performs validation inference, computes the Challenge Metric,
    and runs failure analysis.
    """
    print("\nStarting Validation Evaluation...")
    model = trainer.model
    model.eval()
    device = trainer.device
    val_dataset = trainer.val_dataset

    total_lev_dist = 0
    total_gt_gestures = 0

    # For Failure Analysis
    analysis_data = []

    # Iterate over validation samples
    # Note: val_dataset is a GestureDataset which returns windows.
    # To evaluate full sequences correctly for the metric, we need to reconstruct
    # predictions for the whole sequence.
    # However, GestureDataset is window-based.
    # Strategy: Use the raw data inside the dataset to process full sequences
    # similar to InferenceEngine, but with labels.

    raw_data = val_dataset.raw_data

    with torch.no_grad():
        for i, item in enumerate(raw_data):
            sample_id = item["sample_id"]
            skel = item["skeleton"]
            audio = item["audio"]
            labels_raw = item["labels"]

            # 1. Get Ground Truth Sequence
            gt_sequence = [int(l["id"]) for l in labels_raw]
            # Sort by begin frame just in case
            labels_raw.sort(key=lambda x: x["begin"])
            gt_sequence = [int(l["id"]) for l in labels_raw]

            # 2. Predict Sequence
            # We use sliding window logic similar to InferenceEngine
            features = get_feature_vector(skel, audio, augment=False)
            T = features.shape[0]

            # Sliding window parameters
            window_size = Config.WINDOW_SIZE
            stride = Config.STRIDE_TEST
            num_classes = Config.NUM_CLASSES

            probs_sum = np.zeros((T, num_classes), dtype=np.float32)
            counts = np.zeros((T, 1), dtype=np.float32)

            # Handle short sequences
            if T < window_size:
                pad_len = window_size - T
                padded_feat = np.pad(features, ((0, pad_len), (0, 0)), mode="constant")
                tensor_feat = (
                    torch.from_numpy(padded_feat).float().unsqueeze(0).to(device)
                )

                outputs = model(tensor_feat)
                probs = F.softmax(outputs["logits_s3"], dim=2).cpu().numpy()[0]

                probs_sum += probs[:T]
                counts += 1.0
            else:
                starts = list(range(0, T - window_size + 1, stride))
                if (T - window_size) % stride != 0:
                    starts.append(T - window_size)

                for start in starts:
                    end = start + window_size
                    window_feat = features[start:end]
                    tensor_feat = (
                        torch.from_numpy(window_feat).float().unsqueeze(0).to(device)
                    )

                    outputs = model(tensor_feat)
                    probs = F.softmax(outputs["logits_s3"], dim=2).cpu().numpy()[0]

                    probs_sum[start:end] += probs
                    counts[start:end] += 1.0

            counts[counts == 0] = 1.0
            avg_probs = probs_sum / counts
            pred_labels_frames = np.argmax(avg_probs, axis=1)

            # 3. Decode
            pred_sequence = decode_sequence(pred_labels_frames)

            # 4. Compute Metric
            dist = levenshtein_distance(pred_sequence, gt_sequence)

            total_lev_dist += dist
            total_gt_gestures += len(gt_sequence)

            # Collect data for analysis
            analysis_data.append(
                {
                    "sample_id": sample_id,
                    "lev_dist": dist,
                    "num_frames": T,
                    "num_gt": len(gt_sequence),
                    "error_rate": (
                        dist / len(gt_sequence) if len(gt_sequence) > 0 else 0
                    ),
                }
            )

    # Final Metric
    final_metric = total_lev_dist / total_gt_gestures if total_gt_gestures > 0 else 0.0
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\n=== Failure Analysis ===")
    df_analysis = pd.DataFrame(analysis_data)

    # Correlation
    if not df_analysis.empty:
        corr_len = df_analysis["lev_dist"].corr(df_analysis["num_frames"])
        corr_num = df_analysis["lev_dist"].corr(df_analysis["num_gt"])

        print(f"Correlation (Error vs Sequence Length): {corr_len:.4f}")
        print(f"Correlation (Error vs Num Gestures): {corr_num:.4f}")

        # Identify worst samples
        print("\nTop 3 Worst Samples (Highest Levenshtein Distance):")
        worst = df_analysis.sort_values("lev_dist", ascending=False).head(3)
        print(
            worst[["sample_id", "lev_dist", "num_gt", "num_frames"]].to_string(
                index=False
            )
        )

    return final_metric


# ==========================================
# 4. Main Execution
# ==========================================
def main():
    # 1. Setup
    setup_config()
    print("Configuration setup complete.")

    # 2. Train
    print("Initializing Trainer...")
    trainer = Trainer(load_cached_data=True)
    print("Starting Training...")
    trainer.fit()

    # 3. Validate & Analyze
    metric = evaluate_and_analyze(trainer)

    # 4. Submission
    THRESHOLD = 0.2251
    if metric < THRESHOLD:
        print(
            f"\nValidation Metric ({metric}) is below threshold ({THRESHOLD}). Generating submission..."
        )
        inference_engine = InferenceEngine()
        inference_engine.run_inference(load_cached_data=True)
    else:
        print(
            f"\nValidation Metric ({metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
