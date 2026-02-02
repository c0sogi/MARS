import sys
import os
import torch
import numpy as np
from scipy.stats import pearsonr

# Ensure library is in path
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import (
    set_seed,
    compute_normalized_levenshtein,
    levenshtein_distance,
    decode_predictions_rle,
)
from library.data_loader import GestureDataset, collate_fn
from library.model import GCINet
from library.train import run_training, get_ground_truth_sequence
from library.predict import generate_submission


def main():
    # 1. Setup
    set_seed(Config.SEED)
    Config.setup()

    # 2. Training Phase
    # Increasing epochs to 40 to ensure convergence with augmentation (Cite solution_lesson_node_00102)
    print("Starting Training Phase...")
    run_training(epochs=40, batch_size=Config.BATCH_SIZE, load_cached_data=True)

    # 3. Validation & Failure Analysis Phase
    print("Starting Validation & Failure Analysis...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the best model saved during training
    model = GCINet().to(device)
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Initialize Validation Loader
    val_dataset = GestureDataset(split="val", load_cached_data=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    all_predictions = []
    all_ground_truths = []

    # metrics for failure analysis
    sample_error_rates = []
    sample_input_lengths = []

    with torch.no_grad():
        for batch in val_loader:
            skeleton = batch["skeleton"].to(device)
            audio = batch["audio"].to(device)
            labels = batch["labels"].to(device)
            lengths = batch["lengths"].to(device)
            mask = batch["mask"].to(device)

            # Forward pass
            logits = model(skeleton, audio, lengths, mask)

            batch_size_curr = logits.shape[0]
            for i in range(batch_size_curr):
                length = lengths[i].item()

                # Extract valid sequence
                seq_logits = logits[i, :length, :]
                seq_labels = labels[i, :length]

                # Decode
                pred_seq = decode_predictions_rle(seq_logits)
                gt_seq = get_ground_truth_sequence(seq_labels)

                all_predictions.append(pred_seq)
                all_ground_truths.append(gt_seq)

                # Compute individual error for analysis
                dist = levenshtein_distance(pred_seq, gt_seq)
                gt_len = len(gt_seq)

                # Normalize error by GT length (if 0, use 1.0 if error exists)
                if gt_len > 0:
                    error_rate = dist / gt_len
                else:
                    error_rate = 1.0 if dist > 0 else 0.0

                sample_error_rates.append(error_rate)
                sample_input_lengths.append(length)

    # Compute Final Metric
    final_metric = compute_normalized_levenshtein(all_predictions, all_ground_truths)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation between Input Length and Error Rate
    if len(sample_error_rates) > 1:
        corr, _ = pearsonr(sample_input_lengths, sample_error_rates)
        print(f"Correlation (Input Length vs Error Rate): {corr:.10f}")
    else:
        print("Insufficient samples for correlation analysis.")

    # 4. Submission Phase
    # Condition: Metric must be lower than 0.061224489795918366
    THRESHOLD = 0.061224489795918366

    if final_metric < THRESHOLD:
        print(
            f"Metric {final_metric} meets threshold (< {THRESHOLD}). Generating submission..."
        )
        generate_submission(
            checkpoint_path=checkpoint_path, batch_size=Config.BATCH_SIZE
        )
    else:
        print(
            f"Metric {final_metric} does not meet threshold (< {THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
