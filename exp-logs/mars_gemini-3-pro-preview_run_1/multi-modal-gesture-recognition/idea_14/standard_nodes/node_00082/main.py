import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import from provided library
from library.config import SEED, DEVICE, BACKGROUND_LABEL, BATCH_SIZE
from library.engine import fit, generate_submission, get_dataloaders
from library.model import GCAResNet
from library.metrics import decode_predictions, compute_levenshtein


def set_seed(seed=SEED):
    """Sets the random seed for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_failure_analysis(errors, lengths, num_targets):
    """Computes and prints correlations for failure analysis."""
    print("\n=== Failure Analysis ===")
    if len(errors) < 2:
        print("Not enough samples for correlation analysis.")
        return

    # Correlation with Sequence Length (Frames)
    corr_len, p_len = pearsonr(errors, lengths)
    print(f"Correlation (Error vs Sequence Length): {corr_len:.4f} (p={p_len:.4f})")

    # Correlation with Number of Gestures
    corr_num, p_num = pearsonr(errors, num_targets)
    print(f"Correlation (Error vs Num Gestures): {corr_num:.4f} (p={p_num:.4f})")
    print("========================\n")


def main():
    # 1. Setup
    set_seed()
    print(f"Running on device: {DEVICE}")

    # 2. Training
    # We use 60 epochs which is sufficient for convergence on this small dataset
    # while ensuring the run completes quickly.
    print("Starting training pipeline...")
    best_model_path, stats = fit(num_epochs=60)
    print(f"Training finished. Best model saved at: {best_model_path}")

    # 3. Final Validation & Metric Calculation
    print("Performing final validation evaluation...")

    # Reload the best model
    model = GCAResNet().to(DEVICE)
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.eval()

    # Get Validation Loader (re-using get_dataloaders ensures consistent stats)
    _, val_loader, _ = get_dataloaders(batch_size=BATCH_SIZE)

    total_dist = 0
    total_len_gt = 0

    # Data for failure analysis
    fa_errors = []
    fa_seq_lengths = []
    fa_num_targets = []

    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue

            # Move data to device
            skeletons = batch["skeleton"].to(DEVICE)
            audios = batch["audio"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            lengths = batch["lengths"].to(DEVICE)

            # Inference
            logits = model(skeletons, audios, lengths)

            # Decode Predictions
            preds_batch = decode_predictions(logits, lengths)

            # Process Ground Truth (Remove padding)
            targets_np = labels.cpu().numpy()

            for i, pred_seq in enumerate(preds_batch):
                # Extract clean target sequence
                target_seq = [x for x in targets_np[i] if x != BACKGROUND_LABEL]

                # Compute Metric
                dist = compute_levenshtein(pred_seq, target_seq)

                # Accumulate for global score
                total_dist += dist
                total_len_gt += len(target_seq)

                # Collect for failure analysis
                fa_errors.append(dist)
                fa_seq_lengths.append(lengths[i].item())
                fa_num_targets.append(len(target_seq))

    # Compute Final Metric
    # Avoid division by zero
    denom = max(1, total_len_gt)
    final_metric = total_dist / denom

    # PRINT REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    run_failure_analysis(fa_errors, fa_seq_lengths, fa_num_targets)

    # 5. Submission Generation
    # Threshold defined in task
    THRESHOLD = 0.0824829931972789

    if final_metric < THRESHOLD:
        print(
            f"Metric ({final_metric:.6f}) is below threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(best_model_path, stats)
    else:
        print(
            f"Metric ({final_metric:.6f}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
