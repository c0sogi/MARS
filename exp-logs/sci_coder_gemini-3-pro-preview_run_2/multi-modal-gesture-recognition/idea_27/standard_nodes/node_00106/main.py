import os
import sys
import torch
import numpy as np
from scipy.stats import pearsonr

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, compute_levenshtein_distance
from library.trainer import Trainer
from library.data_loader import GestureDataset, collate_fn
from torch.utils.data import DataLoader


def main():
    # 1. Setup and Initialization
    # Ensure reproducibility and select device
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Invalidate stale cache files to force reprocessing of the full dataset (Cite debug_lesson_4)
    cache_files = [
        os.path.join(Config.WORKING_DIR, "train_data.npz"),
        os.path.join(Config.WORKING_DIR, "val_data.npz"),
        os.path.join(Config.WORKING_DIR, "test_data.npz"),
    ]
    for cf in cache_files:
        if os.path.exists(cf):
            print(f"Removing stale cache file: {cf}")
            os.remove(cf)

    print(f"Initializing Trainer on device: {device}")
    trainer = Trainer(device=device)

    # 2. Training
    # We limit the epochs to 15 to satisfy the "Fast Baseline" requirement while ensuring
    # sufficient convergence on the full dataset.
    print("Starting training pipeline...")
    trainer.fit(epochs=15, patience=5, debug=False)

    # 3. Validation and Failure Analysis
    print("Performing validation assessment and failure analysis...")

    # Load the best model checkpoint saved during training
    checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("Warning: No checkpoint found. Using current model state.")

    trainer.model.eval()

    # Ensure validation loader is available
    if trainer.val_loader is None:
        val_dataset = GestureDataset(split="val", debug=False, load_cached_data=True)
        trainer.val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

    all_preds = []
    all_targets = []
    sample_errors = []
    sample_lengths = []

    # Inference loop on validation set for detailed analysis
    with torch.no_grad():
        for batch in trainer.val_loader:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            lengths = batch["lengths"]

            # Ground Truth: Extract from padded tensor
            batch_targets_cls = batch["targets_cls"].cpu().numpy()

            # Forward Pass
            _, _, out3 = trainer.model(features, mask)
            probs_cls = out3[:, :, : Config.NUM_CLASSES]

            # Decode Predictions
            batch_preds = trainer._decode_predictions(probs_cls, lengths)

            for i in range(len(lengths)):
                length = lengths[i].item()

                # Process Target: Collapse repeats and remove background (0)
                t_seq_frames = batch_targets_cls[i, :length]
                t_seq_collapsed = []
                prev = -1
                for t in t_seq_frames:
                    if t != prev:
                        if t != 0:
                            t_seq_collapsed.append(int(t))
                        prev = t

                pred_seq = batch_preds[i]

                # Compute Metric for this sample
                dist = compute_levenshtein_distance(pred_seq, t_seq_collapsed)

                all_preds.append(pred_seq)
                all_targets.append(t_seq_collapsed)
                sample_errors.append(dist)
                sample_lengths.append(length)

    # Compute Final Validation Metric
    total_distance = sum(sample_errors)
    total_gestures = sum([len(t) for t in all_targets])

    final_metric = total_distance / total_gestures if total_gestures > 0 else 0.0

    # REQUIRED OUTPUT: Print the final validation metric
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation between Error Magnitude and Input Sequence Length
    if len(sample_errors) > 1:
        corr, _ = pearsonr(sample_errors, sample_lengths)
        print(f"Correlation (Error Magnitude vs Input Sequence Length): {corr}")

    # 4. Submission Generation
    # Generate submission only if the metric is lower than the specified threshold
    TARGET_THRESHOLD = 0.06789606035205364

    if final_metric < TARGET_THRESHOLD:
        print(
            f"Metric {final_metric} meets threshold {TARGET_THRESHOLD}. Generating submission..."
        )

        # Predict on Test Set (Trainer handles loading best model)
        predictions = trainer.predict(split="test")

        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

        try:
            with open(submission_path, "w") as f:
                for sample_id, gesture_list in predictions.items():
                    # Format: SessionID,Label1,Label2,...
                    labels_str = ",".join(map(str, gesture_list))
                    f.write(f"{sample_id},{labels_str}\n")
            print(f"Submission saved to {submission_path}")
        except Exception as e:
            print(f"Error writing submission file: {e}")
    else:
        print(
            f"Metric {final_metric} did not meet threshold {TARGET_THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
