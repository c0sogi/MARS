import sys
import os
import torch
import numpy as np
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.utils import seed_everything, levenshtein_distance, load_checkpoint
from library.trainer import Trainer
from library.dataset import create_dataloaders


def main():
    # 1. Setup Configuration and Seeds
    seed_everything(Config.RANDOM_SEED)

    # 2. Training Phase
    print("Initializing training...")
    trainer = Trainer(Config)

    # Train for 35 epochs to ensure quick baseline execution while allowing convergence.
    # The provided Config default is 50, but we reduce slightly to guarantee < 2h runtime.
    trainer.fit(epochs=35)

    # 3. Load Best Model for Evaluation
    # Trainer.fit saves the best model based on validation Loss.
    # We load it to compute the specific Challenge Metric.
    print("Loading best model for validation evaluation...")
    if os.path.exists(Config.MODEL_SAVE_PATH):
        trainer.model, _, _, _ = load_checkpoint(
            trainer.model, None, Config.MODEL_SAVE_PATH
        )
    else:
        print("Warning: No checkpoint found. Using current model state.")

    trainer.model.eval()

    # 4. Validation Inference & Metric Calculation
    print("Computing validation metrics...")
    _, val_loader, _ = create_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    val_lev_distances = []
    val_seq_lengths = []
    val_num_gestures = []

    total_truth_gestures = 0
    total_lev_dist = 0

    with torch.no_grad():
        for features, labels, mask, lengths, ids in val_loader:
            features = features.to(Config.DEVICE)
            mask = mask.to(Config.DEVICE)

            # Forward pass through the model
            outputs = trainer.model(features, mask)
            # Use the output of the final refinement stage for predictions
            final_logits = outputs[-1]
            pred_classes = torch.argmax(final_logits, dim=1)

            # Move data to CPU for processing
            pred_classes_np = pred_classes.cpu().numpy()
            labels_np = labels.cpu().numpy()
            lengths_np = lengths.numpy()

            for i in range(len(ids)):
                length = lengths_np[i]

                # Extract valid sequence (ignoring padding)
                pred_seq_raw = pred_classes_np[i, :length]
                true_seq_raw = labels_np[i, :length]

                # Decode frame-wise predictions into gesture lists
                # This applies RLE and filters out background/short segments
                pred_gestures = trainer._decode_predictions(pred_seq_raw)
                true_gestures = trainer._decode_predictions(true_seq_raw)

                # Compute Levenshtein Distance
                dist = levenshtein_distance(pred_gestures, true_gestures)

                # Store for failure analysis
                val_lev_distances.append(dist)
                val_seq_lengths.append(length)
                val_num_gestures.append(len(true_gestures))

                # Accumulate for global metric
                total_lev_dist += dist
                total_truth_gestures += len(true_gestures)

    # Compute Final Metric: Sum(Distances) / Sum(GroundTruthGestures)
    final_metric = (
        total_lev_dist / total_truth_gestures if total_truth_gestures > 0 else 0.0
    )

    # Print the exact metric as required
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("Performing failure analysis...")
    if len(val_lev_distances) > 1:
        # Correlation: Error vs Sequence Length
        corr_len, _ = pearsonr(val_seq_lengths, val_lev_distances)
        print(f"Correlation (Error vs Seq Length): {corr_len:.4f}")

        # Correlation: Error vs Number of Gestures
        corr_num, _ = pearsonr(val_num_gestures, val_lev_distances)
        print(f"Correlation (Error vs Num Gestures): {corr_num:.4f}")

    # 6. Conditional Submission
    # Threshold specified in the task requirements
    THRESHOLD = 0.32006125574272587

    if final_metric < THRESHOLD:
        print(
            f"Metric ({final_metric}) is below threshold ({THRESHOLD}). Generating submission..."
        )
        # Trainer.predict() loads the best model and writes to submission.csv
        trainer.predict()
    else:
        print(
            f"Metric ({final_metric}) is NOT below threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
