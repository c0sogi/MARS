import os
import sys
import torch
import numpy as np
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

# Ensure library modules are accessible
sys.path.append(os.getcwd())

from library.config import Config
from library.trainer import Trainer
from library.data_loader import MultimodalDataset, collate_fn
from library.utils import decode_predictions, levenshtein_distance


def set_seed(seed):
    """Sets random seeds for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Training
    # Initialize Trainer and start training loop
    # The Trainer handles data loading, model initialization, and the optimization loop.
    trainer = Trainer(device=device)
    trainer.fit()

    # 3. Validation and Failure Analysis
    print("\nStarting Validation and Failure Analysis...")

    # Load the best model saved during training
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        trainer.model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: Best model checkpoint not found. Using current model state.")

    trainer.model.eval()

    # Load Validation Dataset
    val_dataset = MultimodalDataset(mode="val", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=2
    )

    total_dist = 0
    total_ref_len = 0

    # Lists for failure analysis
    sample_errors = []
    sample_lengths = []
    sample_num_gestures = []

    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue

            # Move data to device
            skeleton = batch["skeleton"].to(device)
            audio = batch["audio"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)
            lengths = batch["lengths"].to(device)

            # Inference
            logits = trainer.model(skeleton, audio, mask, lengths)
            preds = torch.argmax(logits, dim=2).cpu().numpy()  # (1, T)
            targets = labels.cpu().numpy()  # (1, T)

            # Get valid sequence length
            length = int(lengths[0].item())

            # Decode sequences
            # Predictions: Smooth -> RLE -> Filter
            pred_seq = decode_predictions(preds[0][:length])

            # Ground Truth: RLE -> Filter (min_len=1 to keep all annotated gestures)
            # We use the same decoding utility but allow shorter segments for GT to be safe
            target_seq = decode_predictions(targets[0][:length], min_len=1)

            # Compute Metric
            dist = levenshtein_distance(pred_seq, target_seq)
            ref_len = len(target_seq)

            total_dist += dist
            total_ref_len += ref_len

            # Store stats
            sample_errors.append(dist)
            sample_lengths.append(length)
            sample_num_gestures.append(ref_len)

    # Compute and Print Final Metric
    final_metric = total_dist / total_ref_len if total_ref_len > 0 else 1.0
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlations
    if len(sample_errors) > 1:
        # Correlation between Error and Sequence Length (frames)
        corr_len, _ = pearsonr(sample_errors, sample_lengths)
        # Correlation between Error and Complexity (Number of Gestures)
        corr_num, _ = pearsonr(sample_errors, sample_num_gestures)

        print("\nFailure Analysis:")
        print(f"Correlation (Error vs Sequence Length): {corr_len:.4f}")
        print(f"Correlation (Error vs Num Gestures): {corr_num:.4f}")

    # 4. Submission
    # Threshold defined in the task
    THRESHOLD = 0.05697278911564626

    if final_metric < THRESHOLD:
        print(
            f"\nMetric {final_metric} is lower than threshold {THRESHOLD}. Generating submission..."
        )
        trainer.predict()
    else:
        print(
            f"\nMetric {final_metric} is not lower than threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
