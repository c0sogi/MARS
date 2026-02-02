import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from nltk import edit_distance

from library.config import Config
from library.utils import (
    set_seed,
    load_checkpoint,
    save_submission,
    compute_levenshtein,
)
from library.data_loader import get_dataloaders
from library.model import DSG_CRCN
from library.train import train_model, decode_sequence


def run():
    # 1. Setup
    Config.setup()
    set_seed(Config.SEED)

    # 2. Train Model (Fast Baseline)
    # We limit epochs to 15 to ensure the script completes quickly within the time limit.
    print("[-] Starting training...")
    trainer = train_model(load_cached_data=True, epochs=15, patience=5)

    # 3. Load Best Model for Evaluation
    # train_model saves the best model to Config.CHECKPOINT_DIR/best_model.pth
    print("[-] Loading best model for evaluation...")
    load_checkpoint(
        trainer.model,
        checkpoint_path=os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"),
    )
    trainer.model.eval()

    # 4. Validation & Failure Analysis
    print("[-] Running validation and failure analysis...")
    val_loader = trainer.val_loader

    all_preds = []
    all_targets = []

    # Lists for failure analysis
    error_magnitudes = []
    seq_lengths = []

    with torch.no_grad():
        for batch in val_loader:
            features = batch[0].to(Config.DEVICE)
            targets = batch[1].to(Config.DEVICE)
            mask = batch[3].to(Config.DEVICE)

            # Forward pass
            outputs = trainer.model(features, mask)
            # Use Stage 3 Class Probabilities for final prediction
            final_cls = outputs["final_cls"]

            # Get hard predictions
            _, pred_indices = torch.max(final_cls, dim=2)

            # Move to CPU for decoding
            pred_indices = pred_indices.cpu().numpy()
            targets_np = targets.cpu().numpy()
            mask_np = mask.cpu().numpy()

            for b in range(features.size(0)):
                # Get valid length of the sequence
                valid_len = np.sum(mask_np[b])

                # Slice valid frames
                p_seq_raw = pred_indices[b, :valid_len]
                t_seq_raw = targets_np[b, :valid_len]

                # Decode sequences (Median Filter + Collapse + Remove Background)
                p_decoded = decode_sequence(p_seq_raw, kernel_size=7)
                t_decoded = decode_sequence(t_seq_raw, kernel_size=1)

                all_preds.append(p_decoded)
                all_targets.append(t_decoded)

                # Calculate per-sample error for failure analysis
                # Error Rate = Levenshtein Distance / Length of Ground Truth
                dist = edit_distance(p_decoded, t_decoded)
                ref_len = len(t_decoded)

                # Avoid division by zero
                err_rate = dist / ref_len if ref_len > 0 else 0.0

                error_magnitudes.append(err_rate)
                seq_lengths.append(valid_len)

    # Compute Final Metric (Global Levenshtein Score)
    final_metric = compute_levenshtein(all_preds, all_targets)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation between Sequence Length and Error Rate
    if len(error_magnitudes) > 1:
        corr, p_value = pearsonr(seq_lengths, error_magnitudes)
        print(
            f"Correlation between Sequence Length and Error Rate: {corr:.4f} (p={p_value:.4f})"
        )
    else:
        print("Not enough samples for correlation analysis.")

    # 5. Submission
    # Generate submission only if metric is better (lower) than threshold
    threshold = 0.06789606035205364

    if final_metric < threshold:
        print(f"[-] Metric {final_metric} < {threshold}. Generating submission...")

        # We need to get the test loader.
        # get_dataloaders returns (train, val, test).
        print("[*] Loading test data...")
        _, _, test_loader = get_dataloaders(load_cached_data=True)

        test_preds = []
        test_ids = []

        trainer.model.eval()

        with torch.no_grad():
            for batch in test_loader:
                features = batch[0].to(Config.DEVICE)
                mask = batch[3].to(Config.DEVICE)
                ids = batch[4]

                outputs = trainer.model(features, mask)
                final_cls = outputs["final_cls"]

                _, pred_indices = torch.max(final_cls, dim=2)
                pred_indices = pred_indices.cpu().numpy()
                mask_np = mask.cpu().numpy()

                for b in range(features.size(0)):
                    valid_len = np.sum(mask_np[b])
                    p_seq_raw = pred_indices[b, :valid_len]

                    # Apply same decoding logic as validation
                    p_decoded = decode_sequence(p_seq_raw, kernel_size=7)

                    test_preds.append(p_decoded)
                    test_ids.append(ids[b])

        # Save submission
        save_submission(test_preds, test_ids)
    else:
        print(f"[-] Metric {final_metric} >= {threshold}. Skipping submission.")


if __name__ == "__main__":
    run()
