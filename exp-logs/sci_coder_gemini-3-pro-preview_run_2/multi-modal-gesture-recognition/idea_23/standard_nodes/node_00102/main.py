import os
import sys
import torch
import numpy as np
from scipy.stats import pearsonr

# 1. Configuration Setup
# We modify Config before importing other modules to ensure settings propagate correctly
from library.config import Config

# Fast Baseline Settings
Config.NUM_EPOCHS = 15  # Sufficient for convergence on this dataset size
Config.BATCH_SIZE = 16  # Optimized for A100
Config.DEBUG_SUBSET_SIZE = (
    None  # Use full dataset for best performance (small enough for <2h)
)
Config.EARLY_STOPPING_PATIENCE = 5

# 2. Imports
from library.engine import Trainer
from library.predict import Predictor
from library.dataset import get_dataloader
from library.utils import set_seed, compute_levenshtein_score, levenshtein_distance


def main():
    # Set reproducibility
    set_seed(Config.SEED)

    print("=== SymG-CRCN Baseline Run ===")
    print(f"Device: {Config.DEVICE}")
    print(f"Epochs: {Config.NUM_EPOCHS}")
    print(f"Batch Size: {Config.BATCH_SIZE}")

    # 3. Training
    print("\n--- Starting Training ---")
    trainer = Trainer()
    trainer.train(num_epochs=Config.NUM_EPOCHS)

    # 4. Validation & Failure Analysis
    print("\n--- Starting Validation & Failure Analysis ---")

    # Load the best model saved during training
    if os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Loading best model from {Config.BEST_MODEL_PATH}")
        trainer.model.load_state_dict(
            torch.load(Config.BEST_MODEL_PATH, map_location=trainer.device)
        )
    else:
        print("Warning: No best model found. Using current model state.")

    trainer.model.eval()

    # Get validation loader
    val_loader = get_dataloader(
        "val", batch_size=Config.BATCH_SIZE, shuffle=False, augment=False
    )

    all_preds = []
    all_targets = []
    all_lengths = []
    sample_errors = []

    with torch.no_grad():
        for batch in val_loader:
            # Move data to device
            features = batch["features"].to(trainer.device)
            mask = batch["mask"].to(trainer.device)
            lengths = batch["lengths"]  # Keep on CPU for decoding
            labels = batch["labels"]  # Keep on CPU for target processing

            # Forward pass
            outputs = trainer.model(features, mask, lengths.to(trainer.device))

            # Use Stage 3 outputs
            s3_logits = outputs["stage3_cls"]

            # Decode predictions
            batch_preds = trainer.decode_batch(s3_logits, lengths)
            all_preds.extend(batch_preds)
            all_lengths.extend(lengths.numpy())

            # Process Ground Truth
            labels_np = labels.numpy()
            for i, seq in enumerate(labels_np):
                length = lengths[i]
                valid_seq = seq[:length]

                # Collapse repeats and remove background (0) and padding (-1)
                collapsed = []
                prev = None
                for label in valid_seq:
                    if label > 0:  # Valid gesture
                        if label != prev:
                            collapsed.append(int(label))
                            prev = label
                    elif label == 0:  # Background
                        prev = 0

                all_targets.append(collapsed)

    # Compute Final Metric
    final_metric = compute_levenshtein_score(all_preds, all_targets)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation between Error and Input Features (Sequence Length)
    # Calculate Levenshtein distance for each sample
    for p, t in zip(all_preds, all_targets):
        dist = levenshtein_distance(p, t)
        sample_errors.append(dist)

    if len(sample_errors) > 1:
        # Correlation between Error Magnitude and Sequence Length
        corr, _ = pearsonr(sample_errors, all_lengths)
        print(f"Correlation between Error and Sequence Length: {corr:.4f}")
    else:
        print("Insufficient data for correlation analysis.")

    # 5. Submission
    # Threshold defined in task
    THRESHOLD = 0.06789606035205364

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        predictor = Predictor()
        predictor.predict()
    else:
        print(
            f"\nMetric ({final_metric}) does not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
