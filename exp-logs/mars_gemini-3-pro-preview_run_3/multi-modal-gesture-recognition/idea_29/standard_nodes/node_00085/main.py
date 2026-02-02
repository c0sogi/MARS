import os
import sys
import random
import numpy as np
import torch
import warnings
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.trainer import Trainer
from library.utils import levenshtein_distance, rle_encode, filter_short_segments

# Suppress warnings
warnings.filterwarnings("ignore")


def set_seeds(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    # 1. Configuration for Fast Baseline
    # Adjust epochs to ensure quick execution while allowing convergence
    Config.NUM_EPOCHS = 35
    Config.BATCH_SIZE = 32

    # Set seeds
    set_seeds(Config.SEED)

    # 2. Initialize Trainer
    # We use the full dataset (debug=False) because the dataset is small (232 samples)
    # and we need sufficient data to meet the metric threshold.
    trainer = Trainer(debug=False)

    # 3. Train the Model
    print("Starting training...")
    trainer.fit()

    # 4. Validation and Failure Analysis
    print("Performing final validation and failure analysis...")

    # Load best model weights
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        trainer.model.load_state_dict(
            torch.load(best_model_path, map_location=trainer.device)
        )
    else:
        print("Warning: Best model checkpoint not found. Using current weights.")

    trainer.model.eval()

    val_errors = []
    val_lengths = []
    val_num_gestures = []

    total_dist = 0
    total_gestures_count = 0

    # Iterate over validation set
    with torch.no_grad():
        for batch in trainer.val_loader:
            # Unpack batch
            features = batch["features"][0].to(trainer.device)  # (T, D)
            labels = batch["labels"][0]  # (T,)

            # Inference using the trainer's sliding window method
            # Accessing protected method as per design pattern for this script
            probs = trainer._sliding_window_inference(features)

            # Decode predictions
            pred_labels = np.argmax(probs, axis=1)
            pred_labels_filtered = filter_short_segments(
                pred_labels, min_duration=Config.MIN_DURATION_FRAMES
            )
            pred_sequence = rle_encode(pred_labels_filtered)

            # Decode ground truth
            gt_sequence = rle_encode(labels.numpy())

            # Calculate Metric (Levenshtein Distance)
            dist = levenshtein_distance(pred_sequence, gt_sequence)

            # Update totals
            total_dist += dist
            total_gestures_count += len(gt_sequence)

            # Collect data for failure analysis
            val_errors.append(dist)
            val_lengths.append(features.shape[0])
            val_num_gestures.append(len(gt_sequence))

    # Compute Final Metric
    final_metric = (
        total_dist / total_gestures_count if total_gestures_count > 0 else 1.0
    )

    # Print required metric format
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation
    if len(val_errors) > 1:
        # Correlation between Error and Sequence Length
        corr_len, _ = pearsonr(val_errors, val_lengths)
        # Correlation between Error and Number of Gestures
        corr_num, _ = pearsonr(val_errors, val_num_gestures)

        print("Failure Analysis Correlations:")
        print(f"  Error vs Sequence Length: {corr_len}")
        print(f"  Error vs Num Gestures: {corr_num}")
    else:
        print("Insufficient validation samples for failure analysis.")

    # 5. Submission Generation
    # Threshold check
    THRESHOLD = 0.2251

    if final_metric < THRESHOLD:
        print(
            f"Metric {final_metric} is lower than threshold {THRESHOLD}. Generating submission..."
        )
        trainer.predict()
    else:
        print(
            f"Metric {final_metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
