import os
import sys
import torch
import pandas as pd
import numpy as np
import scipy.stats
from nltk import edit_distance

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, rle_decode, compute_levenshtein_ratio
from library.trainer import Trainer
from library.data_loader import get_dataloaders


def main():
    # 1. Configuration
    # Increase epochs to ensure convergence (Cite solution_lesson_node_00050)
    Config.NUM_EPOCHS = 50

    # Set random seed for reproducibility
    set_seed(Config.SEED)

    # 2. Train
    # Initialize trainer
    trainer = Trainer()

    # Execute training
    # This will save 'best_model.pth' based on validation score during training
    trainer.train()

    # 3. Final Validation Assessment
    print("\nRunning Final Validation Assessment on the complete validation set...")

    # Get dataloaders
    _, val_loader, _ = get_dataloaders()

    # Load the best model checkpoint
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        trainer.model.load_state_dict(
            torch.load(best_model_path, map_location=trainer.device)
        )
    else:
        print("Warning: Best model checkpoint not found. Using current model state.")

    trainer.model.eval()

    all_preds = []
    all_targets = []

    # Metrics for failure analysis
    sample_errors = []
    sample_lengths = []

    with torch.no_grad():
        for batch in val_loader:
            pose, audio, labels, lengths = batch

            # Move to device
            pose = pose.to(trainer.device)
            audio = audio.to(trainer.device)
            labels = labels.to(trainer.device)

            # Forward pass
            class_logits = trainer.model(pose, audio)

            # Probabilities
            probs = torch.softmax(class_logits, dim=2).cpu().numpy()
            labels_np = labels.cpu().numpy()

            # Process batch
            B = class_logits.shape[0]
            for i in range(B):
                length = lengths[i].item()

                # Slice valid sequence
                valid_probs = probs[i, :length, :]
                valid_targets = labels_np[i, :length]

                # Decode
                pred_seq = rle_decode(
                    valid_probs,
                    min_length=Config.MIN_SEGMENT_LENGTH,
                    background_class=0,
                )
                target_seq = rle_decode(valid_targets, min_length=1, background_class=0)

                all_preds.append(pred_seq)
                all_targets.append(target_seq)

                # Failure Analysis Data Collection
                # 1. Error Magnitude (Levenshtein Distance)
                dist = edit_distance(pred_seq, target_seq)
                sample_errors.append(dist)

                # 2. Sequence Length
                sample_lengths.append(length)

    # 4. Compute and Print Final Metric
    final_metric = compute_levenshtein_ratio(all_preds, all_targets)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    if len(sample_errors) > 1:
        # Correlation: Error vs Sequence Length
        corr_len, p_len = scipy.stats.pearsonr(sample_errors, sample_lengths)
        print(f"Correlation (Error vs Seq Length): {corr_len:.4f} (p={p_len:.4f})")

        # Summary Stats
        print(f"Mean Error: {np.mean(sample_errors):.4f}")
        print(f"Max Error: {np.max(sample_errors)}")
    else:
        print("Not enough samples for correlation analysis.")

    # 6. Conditional Submission
    # Threshold from task description
    THRESHOLD = 0.0824829931972789

    if final_metric < THRESHOLD:
        print(
            f"\nMetric {final_metric} is lower than threshold {THRESHOLD}. Generating submission..."
        )
        trainer.predict()
    else:
        print(
            f"\nMetric {final_metric} is not lower than threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
