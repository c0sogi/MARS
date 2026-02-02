import sys
import os
import numpy as np
import torch

# 1. Patch Config for Speed before importing Trainer
# This ensures the Trainer class picks up the modified configuration
import library.config

from library.utils import set_seed, decode_predictions, levenshtein_distance
from library.trainer import Trainer
from library.config import BACKGROUND_CLASS_ID


def main():
    # 2. Setup
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. Initialize Trainer
    # This will load data (cached) and initialize the model
    trainer = Trainer(device=device)

    # 4. Train
    print("Starting training...")
    trainer.run()

    # 5. Validation Metric
    print("Computing final validation metric...")
    final_score = trainer.validate_metric()
    # Print strictly as required
    print(f"Final Validation Metric: {final_score}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    val_samples = trainer.val_samples
    errors = []
    lengths = []
    num_gestures = []

    # Iterate over validation samples to gather statistics
    print(f"Analyzing {len(val_samples)} validation samples...")
    for sample in val_samples:
        # Run inference
        avg_probs = trainer.run_inference_on_sample(sample)
        pred_seq = decode_predictions(avg_probs)

        # Extract Ground Truth Sequence
        gt_labels = sample["labels"]
        gt_seq = []
        if len(gt_labels) > 0:
            curr = gt_labels[0]
            if curr != BACKGROUND_CLASS_ID:
                gt_seq.append(curr)
            for x in gt_labels[1:]:
                if x != curr:
                    curr = x
                    if curr != BACKGROUND_CLASS_ID:
                        gt_seq.append(curr)

        # Compute Metric for this sample
        dist = levenshtein_distance(pred_seq, gt_seq)

        # Collect stats
        errors.append(dist)
        lengths.append(len(gt_labels))  # Sequence length in frames
        num_gestures.append(len(gt_seq))  # Number of gestures

    # Compute Correlations
    if len(errors) > 1:
        # Use numpy for correlation
        corr_len = np.corrcoef(errors, lengths)[0, 1]
        corr_num = np.corrcoef(errors, num_gestures)[0, 1]

        print(f"Correlation (Error vs Sequence Length): {corr_len:.4f}")
        print(f"Correlation (Error vs Num Gestures): {corr_num:.4f}")
    else:
        print("Insufficient samples for correlation analysis.")

    # 7. Submission
    threshold = 0.2251
    if final_score < threshold:
        print(
            f"\nScore ({final_score}) is lower than threshold ({threshold}). Generating submission..."
        )
        trainer.generate_submission()
    else:
        print(
            f"\nScore ({final_score}) is not lower than threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
