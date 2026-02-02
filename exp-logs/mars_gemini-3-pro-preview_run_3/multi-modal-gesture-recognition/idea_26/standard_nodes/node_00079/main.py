import os
import torch
import numpy as np
from scipy.stats import pearsonr

# 1. Configuration Override for Fast Baseline
from library.config import Config

Config.NUM_EPOCHS = 30
Config.BATCH_SIZE = 32

from library.data_loader import get_data_loaders
from library.trainer import Trainer, set_seed
from library.utils import levenshtein_distance, decode_sequence


def main():
    # Set reproducible seeds
    set_seed(Config.SEED)

    print("Initializing Data Loaders...")
    # train_loader: Windowed data (inputs, targets)
    # val_metric_loader: Full sequence data (inputs, targets, sample_id)
    train_loader, _, val_metric_loader, test_loader = get_data_loaders(
        load_cached_data=True
    )

    print("Initializing Trainer...")
    trainer = Trainer()

    # 2. Train Model
    # We pass val_metric_loader to fit() because Trainer.validate() expects
    # the 3-element tuple structure (features, targets, sid) provided by the full-sequence loader.
    print("Starting Training...")
    trainer.fit(train_loader, val_metric_loader)

    # 3. Final Validation Metric
    print("Computing Final Validation Metric...")
    final_metric = trainer.validate(val_metric_loader)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")
    trainer.model.eval()
    errors = []
    seq_lengths = []
    num_gestures_list = []

    device = trainer.device

    with torch.no_grad():
        for batch in val_metric_loader:
            # Unpack batch (batch_size=1 for full sequences)
            features, targets, _ = batch
            features = features.squeeze(0)  # (Time, Dim)
            targets = targets.squeeze(0).numpy()  # (Time,)

            # Predict
            probs = trainer.predict_sequence(features)
            pred_seq = decode_sequence(probs)

            # Extract target sequence from frame-wise labels (RLE logic)
            target_seq = []
            if len(targets) > 0:
                curr = targets[0]
                for i in range(1, len(targets)):
                    if targets[i] != curr:
                        if curr != 0:
                            target_seq.append(int(curr))
                        curr = targets[i]
                if curr != 0:
                    target_seq.append(int(curr))

            # Compute Metric for this sample
            dist = levenshtein_distance(pred_seq, target_seq)

            # Collect stats
            errors.append(dist)
            seq_lengths.append(features.size(0))
            num_gestures_list.append(len(target_seq))

    # Compute Correlations
    if len(errors) > 1:
        # Correlation with Sequence Length
        corr_len, _ = pearsonr(errors, seq_lengths)
        print(f"Correlation (Error vs Sequence Length): {corr_len:.4f}")

        # Correlation with Number of Gestures
        corr_num, _ = pearsonr(errors, num_gestures_list)
        print(f"Correlation (Error vs Num Gestures): {corr_num:.4f}")
    else:
        print("Insufficient validation samples for correlation analysis.")

    # 5. Submission Generation
    threshold = 0.2251
    if final_metric < threshold:
        print(f"\nMetric {final_metric} < {threshold}. Generating submission...")
        trainer.generate_submission(test_loader, Config.SUBMISSION_PATH)
    else:
        print(
            f"\nMetric {final_metric} >= {threshold}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
