import os
import sys
import torch
import numpy as np
import pandas as pd
import scipy.stats
import nltk

# Monkey-patch configuration for a fast baseline execution
import library.config

library.config.NUM_EPOCHS = 15  # Reduce epochs for speed (default was 100)
library.config.EARLY_STOPPING_PATIENCE = 5  # Reduce patience
library.config.BATCH_SIZE = 32  # Ensure batch size is appropriate

from library.trainer import Trainer
from library.predict import generate_predictions
from library.utils import set_seed
from library.config import SEED, SUBMISSION_FILE


def run_failure_analysis(trainer):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between error magnitude and input features.
    """
    print("\n=== Starting Failure Analysis ===")
    trainer.model.eval()

    errors = []
    seq_lengths = []
    num_gestures = []

    device = trainer.device

    with torch.no_grad():
        for batch_data in trainer.val_loader:
            features, cls_targets, bnd_targets, lengths, mask, _ = batch_data

            features = features.to(device)
            mask = mask.to(device)

            # Forward pass
            outputs = trainer.model(features, mask)

            # Get Stage 3 logits: (N, L, C) -> Permute to (N, C, L) for decoding logic if needed,
            # but trainer.decode_batch expects (N, C, L) based on my reading of the provided trainer code?
            # Let's check trainer.py provided in prompt.
            # In trainer.py: validate_epoch does: s3_logits = outputs["stage3"]["cls"] (N, L, C)
            # then s3_logits_permuted = s3_logits.permute(0, 2, 1) (N, C, L)
            # then decode_batch(s3_logits_permuted, lengths)

            s3_logits = outputs["stage3"]["cls"]  # (N, L, C)
            s3_logits_permuted = s3_logits.permute(0, 2, 1)  # (N, C, L)

            # Decode
            batch_preds = trainer.decode_batch(s3_logits_permuted, lengths)
            batch_truths = trainer.get_truth_sequences(cls_targets, lengths)

            for i in range(len(batch_preds)):
                pred = batch_preds[i]
                truth = batch_truths[i]
                length = lengths[i]

                # Calculate Levenshtein distance for this sample
                dist = nltk.edit_distance(pred, truth)

                # Normalize error by truth length (if > 0) to get a rate, or just use raw distance
                # The metric is total_dist / total_truth_len.
                # For correlation, raw distance or rate per sample is fine. Let's use raw distance.

                errors.append(dist)
                seq_lengths.append(
                    length.item() if isinstance(length, torch.Tensor) else length
                )
                num_gestures.append(len(truth))

    # Convert to numpy for stats
    errors = np.array(errors)
    seq_lengths = np.array(seq_lengths)
    num_gestures = np.array(num_gestures)

    # Correlation Analysis
    if len(errors) > 1:
        corr_len, _ = scipy.stats.pearsonr(errors, seq_lengths)
        corr_num, _ = scipy.stats.pearsonr(errors, num_gestures)

        print(f"Correlation (Error vs Sequence Length): {corr_len:.4f}")
        print(f"Correlation (Error vs Num Gestures): {corr_num:.4f}")

        # Identify high error cases
        high_error_indices = np.argsort(errors)[-5:]
        print("\nTop 5 High Error Samples (Indices in Val Set):")
        for idx in high_error_indices:
            print(
                f"  Idx: {idx}, Error: {errors[idx]}, Len: {seq_lengths[idx]}, Gestures: {num_gestures[idx]}"
            )
    else:
        print("Not enough samples for correlation analysis.")

    print("=== Failure Analysis Complete ===\n")


def main():
    # 1. Set Seed
    set_seed(SEED)

    print("Initializing Training Pipeline...")

    # 2. Initialize Trainer
    # We use the default subset_size=None to train on full data,
    # relying on reduced epochs (monkey-patched above) for speed.
    trainer = Trainer(subset_size=None)

    # 3. Train Model
    print("Starting Training...")
    trainer.train()

    # 4. Report Final Metric
    # The trainer updates self.best_val_score with the lowest Levenshtein error rate
    final_metric = trainer.best_val_score
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    run_failure_analysis(trainer)

    # 6. Submission
    # Threshold from task description
    THRESHOLD = 0.06789606035205364

    if final_metric < THRESHOLD:
        print(
            f"Validation metric ({final_metric}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        generate_predictions(
            checkpoint_path=os.path.join(library.config.WORKING_DIR, "best_model.pth"),
            output_file=SUBMISSION_FILE,
        )
    else:
        print(
            f"Validation metric ({final_metric}) does not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
