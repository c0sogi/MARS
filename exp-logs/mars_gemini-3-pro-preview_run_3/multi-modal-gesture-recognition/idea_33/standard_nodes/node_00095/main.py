import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr
from itertools import groupby

# 1. Configuration & Patching for Fast Baseline
import library.config
import library.trainer

# Patch the number of epochs to ensure fast execution (Fast Baseline)
library.trainer.NUM_EPOCHS = 15
library.config.NUM_EPOCHS = 15
# Reduce patience to fail fast if no improvement
library.trainer.EARLY_STOPPING_PATIENCE = 5

from library.config import (
    SEED,
    MODEL_SAVE_PATH,
    BATCH_SIZE,
    WINDOW_SIZE,
    NUM_CLASSES,
    VAL_METADATA_PATH,
)
from library.trainer import ModelTrainer
from library.inference import SequencePredictor
from library.data_loader import get_data_loaders
from library.utils import seed_everything, levenshtein_distance, filter_short_segments
from library.model import RGHCMN


def main():
    # Set seeds for reproducibility
    seed_everything(SEED)

    print("========================================")
    print("      RG-HCMN Fast Baseline Run         ")
    print("========================================")

    # ---------------------------------------------------------
    # 2. Training
    # ---------------------------------------------------------
    print("\n[Step 1/4] Starting Training...")
    trainer = ModelTrainer()
    trainer.train()

    # ---------------------------------------------------------
    # 3. Validation & Metric Calculation
    # ---------------------------------------------------------
    print("\n[Step 2/4] Performing Final Validation...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the best model saved by the trainer
    model = RGHCMN().to(device)
    if os.path.exists(MODEL_SAVE_PATH):
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded best model from {MODEL_SAVE_PATH}")
    else:
        print("Error: Model file not found. Training might have failed.")
        return

    model.eval()

    # Get validation loader
    _, val_loader, _ = get_data_loaders(batch_size=BATCH_SIZE)
    val_dataset = val_loader.dataset

    # Buffers for sequence reconstruction
    sample_probs = {}
    sample_counts = {}

    # Initialize buffers
    for i, sample_id in enumerate(val_dataset.sample_ids):
        start, end = val_dataset.sample_boundaries[i]
        length = end - start
        sample_probs[i] = torch.zeros(
            (length, NUM_CLASSES), device="cpu", dtype=torch.float
        )
        sample_counts[i] = torch.zeros((length, 1), device="cpu", dtype=torch.float)

    # Inference Loop
    with torch.no_grad():
        for batch_x, _, batch_indices, batch_starts in val_loader:
            batch_x = batch_x.to(device)

            # Forward pass
            outputs = model(batch_x)
            logits = outputs["logits_3"]  # Use Stage 3 output
            probs = F.softmax(logits, dim=2).cpu()

            # Accumulate
            for k in range(len(batch_indices)):
                s_idx = batch_indices[k].item()
                r_start = batch_starts[k].item()

                total_len = sample_probs[s_idx].shape[0]
                valid_len = min(WINDOW_SIZE, total_len - r_start)

                sample_probs[s_idx][r_start : r_start + valid_len] += probs[
                    k, :valid_len, :
                ]
                sample_counts[s_idx][r_start : r_start + valid_len] += 1.0

    # Process results
    all_preds = []
    all_gts = []
    sample_errors = []

    # Meta-features for Failure Analysis
    meta_duration = []
    meta_num_gestures = []
    meta_kinematic_energy = []

    total_dist = 0
    total_gestures_count = 0

    print("Computing metrics per sample...")
    for i in range(len(val_dataset.sample_ids)):
        # 1. Reconstruct Prediction
        counts = sample_counts[i]
        counts[counts == 0] = 1.0
        avg_probs = sample_probs[i] / counts
        frame_preds = torch.argmax(avg_probs, dim=1).numpy()
        pred_seq = filter_short_segments(frame_preds)
        all_preds.append(pred_seq)

        # 2. Get Ground Truth
        global_start, global_end = val_dataset.sample_boundaries[i]
        dense_labels = val_dataset.all_labels[global_start:global_end]
        gt_seq = [k for k, g in groupby(dense_labels) if k != 0]
        all_gts.append(gt_seq)

        # 3. Compute Error
        dist = levenshtein_distance(pred_seq, gt_seq)
        sample_errors.append(dist)
        total_dist += dist
        total_gestures_count += len(gt_seq)

        # 4. Extract Meta Features
        duration = global_end - global_start
        meta_duration.append(duration)
        meta_num_gestures.append(len(gt_seq))

        # Simple proxy for kinematic energy: mean velocity magnitude
        # We need to access the raw skeleton data from the dataset
        if val_dataset.all_skeletons is not None:
            skel_segment = val_dataset.all_skeletons[
                global_start:global_end
            ]  # (T, J, 3)
            if len(skel_segment) > 1:
                # Velocity: diff between frames
                vel = np.diff(skel_segment, axis=0)
                # Magnitude per joint per frame
                vel_mag = np.linalg.norm(vel, axis=2)
                # Mean over all joints and frames
                energy = np.mean(vel_mag)
                meta_kinematic_energy.append(energy)
            else:
                meta_kinematic_energy.append(0.0)
        else:
            meta_kinematic_energy.append(0.0)

    # Calculate Final Metric
    final_metric = (
        total_dist / total_gestures_count if total_gestures_count > 0 else float("inf")
    )

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # ---------------------------------------------------------
    # 4. Failure Analysis
    # ---------------------------------------------------------
    print("\n[Step 3/4] Failure Analysis...")

    errors_arr = np.array(sample_errors)
    duration_arr = np.array(meta_duration)
    gestures_arr = np.array(meta_num_gestures)
    energy_arr = np.array(meta_kinematic_energy)

    # Correlations
    if len(errors_arr) > 1:
        # Avoid constant input warnings if variance is 0
        if np.std(errors_arr) > 0 and np.std(duration_arr) > 0:
            corr_dur, _ = pearsonr(errors_arr, duration_arr)
            print(f"Correlation (Error vs Duration): {corr_dur:.4f}")

        if np.std(errors_arr) > 0 and np.std(gestures_arr) > 0:
            corr_gest, _ = pearsonr(errors_arr, gestures_arr)
            print(f"Correlation (Error vs Num Gestures): {corr_gest:.4f}")

        if np.std(errors_arr) > 0 and np.std(energy_arr) > 0:
            corr_energy, _ = pearsonr(errors_arr, energy_arr)
            print(f"Correlation (Error vs Kinematic Energy): {corr_energy:.4f}")

        # Top Failures
        worst_indices = np.argsort(errors_arr)[-5:][::-1]
        print("\nTop 5 High Error Samples:")
        for idx in worst_indices:
            sid = val_dataset.sample_ids[idx]
            print(
                f"  Sample: {sid} | Error: {errors_arr[idx]} | GT Len: {gestures_arr[idx]} | Pred Len: {len(all_preds[idx])}"
            )
    else:
        print("Not enough samples for correlation analysis.")

    # ---------------------------------------------------------
    # 5. Submission
    # ---------------------------------------------------------
    print("\n[Step 4/4] Checking Submission Criteria...")
    THRESHOLD = 0.2251

    if final_metric < THRESHOLD:
        print(
            f"Metric {final_metric:.5f} is better than threshold {THRESHOLD}. Generating submission..."
        )
        predictor = SequencePredictor()
        predictor.run()
    else:
        print(
            f"Metric {final_metric:.5f} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
