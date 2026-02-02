import os
import sys
import numpy as np
import pandas as pd
import torch
import scipy.stats
import random

# Import provided library modules
from library import config, utils, data_loader, model, trainer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    # 1. Initialization
    set_seed(config.SEED)
    print("Initializing SH-PAM-CN Pipeline...")

    # Define fast run parameters
    # The dataset is small (232 train samples), so 25 epochs is sufficient for a baseline
    # and fits well within the time limit.
    FAST_EPOCHS = 25

    # Initialize Trainer
    # We use the default device detection logic inside Trainer
    pipeline = trainer.Trainer()

    # 2. Training
    print(f"Starting training for {FAST_EPOCHS} epochs...")
    pipeline.run_training(epochs=FAST_EPOCHS)

    # 3. Validation & Metric Reporting
    print("Performing final validation...")

    # Load best model for validation
    if os.path.exists(pipeline.best_model_path):
        pipeline.model.load_state_dict(
            torch.load(pipeline.best_model_path, map_location=pipeline.device)
        )
    pipeline.model.eval()

    # Get val loader
    _, val_loader, _ = pipeline.get_dataloaders()

    # Load GT
    val_metadata = pd.read_csv(config.VAL_METADATA_PATH)
    gt_dict = utils.parse_ground_truth(val_metadata)

    predictions_dict = {}
    sample_errors = []
    sample_lengths = []
    sample_complexity = []  # Number of GT gestures

    with torch.no_grad():
        for features, _, sample_id_tuple in val_loader:
            sample_id = sample_id_tuple[0]
            features = features.to(pipeline.device)

            # Forward
            outputs = pipeline.model(features)
            final_logits = outputs[-1]

            # Decode
            probs = torch.softmax(final_logits, dim=2)
            preds = torch.argmax(probs, dim=2).squeeze(0).cpu().numpy()

            predicted_labels = utils.decode_predictions_to_labels(preds)
            predictions_dict[sample_id] = predicted_labels

            # For Failure Analysis
            gt_labels = gt_dict.get(sample_id, [])
            dist = utils.levenshtein_distance(predicted_labels, gt_labels)

            sample_errors.append(dist)
            sample_lengths.append(features.shape[1])  # Time dimension
            sample_complexity.append(len(gt_labels))

    # Compute Final Metric
    # Metric = Total Distance / Total GT Gestures
    total_distance = sum(sample_errors)
    total_gt = sum(sample_complexity)

    final_metric = total_distance / total_gt if total_gt > 0 else 0.0

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")
    if len(sample_errors) > 1:
        # Correlation: Error Magnitude vs Input Features
        # We normalize error by sequence length or complexity?
        # The metric is unnormalized distance, but for correlation, raw distance is fine to see if longer seqs fail more.

        corr_len, _ = scipy.stats.pearsonr(sample_lengths, sample_errors)
        corr_comp, _ = scipy.stats.pearsonr(sample_complexity, sample_errors)

        print(f"Correlation (Error vs Sequence Length): {corr_len:.4f}")
        print(f"Correlation (Error vs Num Gestures): {corr_comp:.4f}")

        if corr_len > 0.5:
            print(
                "Observation: High correlation with sequence length. Long sequences are prone to cumulative errors."
            )
        if corr_comp > 0.5:
            print(
                "Observation: High correlation with complexity. Dense gesture sequences are harder to recognize."
            )
    else:
        print("Insufficient validation samples for correlation analysis.")

    # 5. Submission
    THRESHOLD = 0.1860643185298622

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        pipeline.predict_test()
    else:
        print(
            f"\nMetric ({final_metric}) did not beat threshold ({THRESHOLD}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
