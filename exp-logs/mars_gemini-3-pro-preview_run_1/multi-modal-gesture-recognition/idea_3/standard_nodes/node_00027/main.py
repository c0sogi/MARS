import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import (
    set_seed,
    decode_predictions,
    compute_normalized_levenshtein,
    levenshtein_distance,
)
from library.data_loader import GestureDataset, collate_fn
from library.model import MSDIGModel
from library.train import train_model, extract_ground_truth_sequence


def run_detailed_validation(model, device):
    """
    Runs inference on the validation set to compute the final metric
    and gather data for failure analysis.
    """
    val_dataset = GestureDataset(split="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    model.eval()

    results = []
    all_preds = []
    all_targets = []

    print("Running detailed validation inference...")

    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue

            skeleton = batch["skeleton"].to(device)
            audio = batch["audio"].to(device)
            # labels kept on CPU for GT extraction
            labels = batch["labels"]
            lengths = batch["lengths"].to(device)
            sample_ids = batch["sample_ids"]

            # Forward Pass
            logits = model(skeleton, audio, lengths)

            # Decode
            batch_preds = decode_predictions(logits)

            # Process batch
            for i in range(len(sample_ids)):
                sid = sample_ids[i]
                length = lengths[i].item()

                # Ground Truth
                seq_labels = labels[i, :length]
                gt_seq = extract_ground_truth_sequence(seq_labels)

                # Prediction
                pred_seq = batch_preds[i]

                all_preds.append(pred_seq)
                all_targets.append(gt_seq)

                # Metrics for Failure Analysis
                dist = levenshtein_distance(pred_seq, gt_seq)

                # Input Features
                # Skeleton variance (proxy for movement amount)
                skel_data = skeleton[i, :length].cpu().numpy()
                skel_var = np.var(skel_data)

                # Audio energy (proxy for sound intensity)
                audio_data = audio[i, :length].cpu().numpy()
                audio_energy = np.mean(audio_data)

                results.append(
                    {
                        "sample_id": sid,
                        "levenshtein_dist": dist,
                        "seq_length": length,
                        "num_gestures_gt": len(gt_seq),
                        "skel_variance": skel_var,
                        "audio_energy": audio_energy,
                    }
                )

    # Compute Final Metric
    final_metric = compute_normalized_levenshtein(all_preds, all_targets)

    return final_metric, pd.DataFrame(results)


def generate_submission(model, device):
    """
    Generates predictions for the test set and saves to CSV.
    """
    print("Generating submission for test set...")

    test_dataset = GestureDataset(split="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    model.eval()
    submission_lines = []

    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue

            skeleton = batch["skeleton"].to(device)
            audio = batch["audio"].to(device)
            lengths = batch["lengths"].to(device)
            sample_ids = batch["sample_ids"]

            logits = model(skeleton, audio, lengths)
            batch_preds = decode_predictions(logits)

            for i, sid in enumerate(sample_ids):
                pred_seq = batch_preds[i]
                # Format: SessionID,label1,label2,...
                pred_str = ",".join(map(str, pred_seq))
                line = f"{sid},{pred_str}"
                submission_lines.append(line)

    # Save to file
    with open(Config.SUBMISSION_PATH, "w") as f:
        for line in submission_lines:
            f.write(line + "\n")

    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def main():
    # 1. Setup
    Config.setup_environment()
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Train
    # Using 25 epochs to ensure convergence within the time limit.
    # The dataset is small, so this is very fast.
    print("--- Starting Training ---")
    train_model(limit=None, epochs=25)

    # 3. Load Best Model
    print("--- Loading Best Model ---")
    model = MSDIGModel().to(device)
    if not os.path.exists(Config.BEST_MODEL_PATH):
        print("Error: Best model file not found.")
        return

    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    # 4. Detailed Validation & Failure Analysis
    print("--- Performing Validation & Failure Analysis ---")
    final_metric, analysis_df = run_detailed_validation(model, device)

    # REQUIRED PRINT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation
    if not analysis_df.empty:
        print("\nFailure Analysis (Correlation with Levenshtein Distance):")
        features = ["seq_length", "num_gestures_gt", "skel_variance", "audio_energy"]
        for feat in features:
            if feat in analysis_df.columns:
                # Handle cases with NaN or constant values
                if analysis_df[feat].std() == 0:
                    corr = 0.0
                else:
                    corr, _ = pearsonr(
                        analysis_df["levenshtein_dist"], analysis_df[feat]
                    )
                print(f"  Correlation with {feat}: {corr:.4f}")

    # 5. Submission
    # Threshold check
    THRESHOLD = 0.1292517006802721

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission."
        )
        generate_submission(model, device)
    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
