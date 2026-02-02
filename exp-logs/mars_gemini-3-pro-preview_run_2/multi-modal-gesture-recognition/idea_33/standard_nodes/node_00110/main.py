import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import from the provided library
from library.config import Config
from library.train import Trainer
from library.utils import (
    compute_normalized_levenshtein,
    post_process_and_decode,
    decode_sequence,
    levenshtein_distance,
)


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Override Config for Fast Baseline and Path Requirements
    Config.NUM_EPOCHS = 10  # Reduced for fast baseline execution
    Config.SUBMISSION_DIR = "./submission"

    # Ensure directories exist
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Set seeds for reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)

    print("Configuration configured for fast baseline.")
    print(f"Epochs: {Config.NUM_EPOCHS}")
    print(f"Device: {Config.DEVICE}")

    # -------------------------------------------------------------------------
    # 2. Training
    # -------------------------------------------------------------------------
    trainer = Trainer(debug=False)
    trainer.fit()

    # -------------------------------------------------------------------------
    # 3. Validation & Metric Extraction
    # -------------------------------------------------------------------------
    # Load the best model state
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        trainer.model.load_state_dict(
            torch.load(best_model_path, map_location=trainer.device)
        )
        print(f"Loaded best model from {best_model_path}")
    else:
        print("Warning: No checkpoint found. Using current model state.")

    # Run validation to get the final metric
    # We use the trainer's validate method but we need the exact value
    _, final_metric = trainer.validate(epoch_idx=Config.NUM_EPOCHS)

    # Print the required metric format
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 4. Failure Analysis
    # -------------------------------------------------------------------------
    print("\nRunning Failure Analysis on Validation Set...")
    trainer.model.eval()

    val_errors = []
    val_seq_lengths = []
    val_num_gestures = []

    with torch.no_grad():
        for features, targets, mask, _ in trainer.val_loader:
            features = features.to(trainer.device)
            targets = targets.to(trainer.device)
            mask = mask.to(trainer.device)

            outputs = trainer.model(features, mask)
            final_output = outputs[-1]
            cls_logits = final_output["cls"]

            batch_size = features.size(0)
            for b in range(batch_size):
                valid_len = int(mask[b].sum().item())

                # Predictions
                sample_logits = cls_logits[b, :valid_len, :].cpu().numpy()
                pred_sequence = post_process_and_decode(
                    sample_logits,
                    kernel_size=7,
                    background_class_id=Config.BACKGROUND_CLASS_ID,
                )

                # Ground Truth
                sample_targets = targets[b, :valid_len].cpu().numpy()
                gt_sequence = decode_sequence(
                    sample_targets, background_class_id=Config.BACKGROUND_CLASS_ID
                )

                # Compute Error
                dist = levenshtein_distance(pred_sequence, gt_sequence)

                # Collect Metrics
                val_errors.append(dist)
                val_seq_lengths.append(valid_len)
                val_num_gestures.append(len(gt_sequence))

    # Compute Correlations
    if len(val_errors) > 1:
        corr_len, _ = pearsonr(val_errors, val_seq_lengths)
        corr_num, _ = pearsonr(val_errors, val_num_gestures)

        print(f"Correlation (Error vs Sequence Length): {corr_len:.4f}")
        print(f"Correlation (Error vs Num Gestures): {corr_num:.4f}")

        if abs(corr_len) > 0.3:
            print(
                "Observation: Model performance degrades significantly with sequence length."
            )
        if abs(corr_num) > 0.3:
            print(
                "Observation: Model struggles with sequences containing many gestures."
            )
    else:
        print("Insufficient validation samples for correlation analysis.")

    # -------------------------------------------------------------------------
    # 5. Submission
    # -------------------------------------------------------------------------
    THRESHOLD = 0.06789606035205364

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.predict()

        submission_file = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        if os.path.exists(submission_file):
            print(f"Submission generated successfully at {submission_file}")
        else:
            print("Error: Submission file was not created.")
    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
