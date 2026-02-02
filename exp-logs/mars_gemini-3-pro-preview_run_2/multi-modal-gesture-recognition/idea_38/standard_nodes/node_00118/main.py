import sys
import os
import torch
import numpy as np
import nltk
from scipy.stats import pearsonr

# Import from the provided library files
from library.config import Config, set_seed
from library.trainer import Trainer
from library.utils import decode_predictions, median_filter_predictions


def main():
    # 1. Setup and Reproducibility
    set_seed(Config.SEED)

    # 2. Initialize Trainer
    # This sets up the HCRG-CN model, loss, optimizer, and data loaders
    trainer = Trainer()

    # 3. Training Loop
    # We use a reasonable number of epochs (25) with early stopping to ensure
    # the model converges enough to meet the strict metric threshold while remaining fast.
    print("Starting training...")
    trainer.fit(epochs=25, early_stopping_patience=7)

    # 4. Report Final Metric
    # This is required for the task evaluation
    print(f"Final Validation Metric: {trainer.best_val_score}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis on Validation Set...")

    # Load the best model weights to ensure analysis reflects the reported metric
    if os.path.exists(trainer.checkpoint_path):
        trainer.model.load_state_dict(
            torch.load(trainer.checkpoint_path, map_location=trainer.device)
        )

    trainer.model.eval()

    val_errors = []
    val_lengths = []

    # Iterate over validation set to compute per-sample metrics
    with torch.no_grad():
        for batch in trainer.val_loader:
            features, labels, boundaries, mask, lengths = batch

            features = features.to(trainer.device)
            mask = mask.to(trainer.device)

            # Forward pass
            predictions = trainer.model(features, mask)

            # Use Stage 3 predictions for final analysis
            s3_cls_probs, _ = predictions["stage3"]
            predicted_indices = torch.argmax(s3_cls_probs, dim=2).cpu().numpy()

            # Apply median filtering (same as inference pipeline)
            filtered_indices = median_filter_predictions(predicted_indices)

            # Get ground truth
            batch_labels = labels.cpu().numpy()
            batch_lengths = lengths.cpu().numpy()

            for i in range(len(features)):
                length = batch_lengths[i]

                # Decode to gesture sequences
                pred_seq = decode_predictions(filtered_indices[i, :length])
                true_seq = decode_predictions(batch_labels[i, :length])

                # Compute Levenshtein distance for this specific sample
                dist = nltk.edit_distance(pred_seq, true_seq)

                val_errors.append(dist)
                val_lengths.append(length)

    # Compute and print correlation
    if len(val_errors) > 1:
        corr, _ = pearsonr(val_errors, val_lengths)
        print(
            f"Correlation between Error Magnitude (Levenshtein Distance) and Sequence Length: {corr}"
        )
    else:
        print("Insufficient data for correlation analysis.")

    # 6. Submission Generation
    # Strict threshold check as per task requirements
    THRESHOLD = 0.06789606035205364

    if trainer.best_val_score < THRESHOLD:
        print(
            f"\nValidation metric ({trainer.best_val_score}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.predict()
    else:
        print(
            f"\nValidation metric ({trainer.best_val_score}) does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
