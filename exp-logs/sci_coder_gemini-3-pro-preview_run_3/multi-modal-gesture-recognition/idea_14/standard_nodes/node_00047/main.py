import sys
import os
import shutil
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import from the provided library files
from library.config import Config
from library.trainer import Trainer, set_seed
from library.inference import InferenceEngine
from library.dataset import GestureDataset
from library.utils import run_length_encoding, compute_levenshtein


def main():
    # ==========================================
    # 1. Setup & Configuration
    # ==========================================
    # Set seeds for reproducibility
    set_seed(Config.SEED)

    # Cite debug_lesson_2: Invalidate cache to force re-processing with fixed robust_load_mat
    if os.path.exists(Config.CACHE_DIR):
        print(f"Clearing cache at {Config.CACHE_DIR} to ensure data integrity...")
        shutil.rmtree(Config.CACHE_DIR)

    # Ensure working directories exist
    Config.setup()

    print(f"Running on device: {Config.DEVICE}")
    print(f"Configured Epochs: {Config.NUM_EPOCHS}")

    # ==========================================
    # 2. Training
    # ==========================================
    print("\n=== Starting Training Phase ===")
    trainer = Trainer()
    trainer.fit()
    print("Training phase completed.")

    # ==========================================
    # 3. Validation & Failure Analysis
    # ==========================================
    print("\n=== Starting Validation & Failure Analysis ===")

    # Load the best model saved during training
    # We use InferenceEngine which loads Config.MODEL_SAVE_PATH by default
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print(
            f"Error: Model file not found at {Config.MODEL_SAVE_PATH}. Training might have failed."
        )
        return

    inference_engine = InferenceEngine()

    # Load Validation Dataset
    val_dataset = GestureDataset(split="val", mode="inference", load_cached_data=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    total_distance = 0
    total_gestures = 0
    sample_stats = []

    # Evaluation Loop
    inference_engine.model.eval()
    with torch.no_grad():
        for features, labels, sample_id_tuple in val_loader:
            sample_id = sample_id_tuple[0]

            # Move data to device
            features = features.squeeze(0).to(Config.DEVICE)
            labels = labels.squeeze(0).numpy()

            # Predict
            probs = inference_engine.predict_sliding_window(features)
            pred_labels = torch.argmax(probs, dim=1).cpu().numpy()

            # Decode sequences
            pred_seq = run_length_encoding(pred_labels)
            gt_seq = run_length_encoding(labels)

            # Compute Metric
            dist = compute_levenshtein(pred_seq, gt_seq)
            n_gestures = len(gt_seq)

            total_distance += dist
            total_gestures += n_gestures

            # Store statistics for failure analysis
            sample_stats.append(
                {
                    "sample_id": sample_id,
                    "levenshtein_dist": dist,
                    "seq_len": features.shape[0],
                    "num_gestures": n_gestures,
                }
            )

    # Compute Final Metric
    final_metric = total_distance / total_gestures if total_gestures > 0 else 0.0

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    df_stats = pd.DataFrame(sample_stats)
    if not df_stats.empty:
        print("\n--- Failure Analysis Report ---")

        # Correlation: Error vs Sequence Length
        if df_stats["seq_len"].std() > 0 and df_stats["levenshtein_dist"].std() > 0:
            corr_len, _ = pearsonr(df_stats["levenshtein_dist"], df_stats["seq_len"])
            print(f"Correlation (Error vs Seq Length): {corr_len:.4f}")
        else:
            print("Correlation (Error vs Seq Length): Undefined (zero variance)")

        # Correlation: Error vs Num Gestures
        if (
            df_stats["num_gestures"].std() > 0
            and df_stats["levenshtein_dist"].std() > 0
        ):
            corr_num, _ = pearsonr(
                df_stats["levenshtein_dist"], df_stats["num_gestures"]
            )
            print(f"Correlation (Error vs Num Gestures): {corr_num:.4f}")
        else:
            print("Correlation (Error vs Num Gestures): Undefined (zero variance)")

        # Worst Samples
        print("\nSamples with Highest Error:")
        print(
            df_stats.sort_values("levenshtein_dist", ascending=False)
            .head(5)
            .to_string(index=False)
        )

    # ==========================================
    # 4. Submission Generation
    # ==========================================
    THRESHOLD = 0.2251

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric:.5f}) is below threshold ({THRESHOLD}). Generating submission..."
        )
        inference_engine.generate_submission()
    else:
        print(
            f"\nMetric ({final_metric:.5f}) is NOT below threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
