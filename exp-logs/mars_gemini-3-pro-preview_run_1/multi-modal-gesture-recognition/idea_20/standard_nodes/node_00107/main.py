import sys
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Ensure library imports work by adding current directory to path
sys.path.append(os.getcwd())

from library.config import Config, set_seed
from library.data_loader import GestureDataset, collate_fn
from library.model import DAGINet
from library.train import run_training
from library.utils import (
    compute_levenshtein_score,
    levenshtein_distance,
    apply_median_filter,
    rle_decode,
    predictions_to_submission_format,
)


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Train
    # Execute training pipeline using the provided library function
    # Config.NUM_EPOCHS (60) is appropriate for a fast baseline on this dataset size
    print("Starting training pipeline...")
    run_training(
        num_epochs=Config.NUM_EPOCHS,
        batch_size=Config.BATCH_SIZE,
        load_cached_data=True,
    )

    # 3. Validation & Evaluation
    print("Loading best model for validation evaluation...")
    model = DAGINet().to(device)

    # Load the best checkpoint saved during training
    if not os.path.exists(Config.BEST_MODEL_PATH):
        print("Error: Best model checkpoint not found.")
        return

    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    val_dataset = GestureDataset(split="val", load_cached_data=True, transform=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    all_preds = []
    all_truths = []
    sample_errors = []

    # Features for failure analysis
    feat_seq_len = []
    feat_num_gestures = []

    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue

            skeleton = batch["skeleton"].to(device)
            audio = batch["audio"].to(device)
            labels = batch["labels"].to(device)
            lengths = batch["lengths"].to(device)

            logits = model(skeleton, audio, lengths)
            preds = torch.argmax(logits, dim=2)

            for i in range(len(lengths)):
                valid_len = lengths[i].item()
                raw_pred = preds[i, :valid_len].cpu().numpy()
                raw_true = labels[i, :valid_len].cpu().numpy()

                # Decode Predictions
                smoothed = apply_median_filter(raw_pred, kernel_size=5)
                pred_seq = rle_decode(
                    smoothed, min_duration=5, background_id=Config.BACKGROUND_CLASS_ID
                )

                # Decode Ground Truth
                true_seq = rle_decode(
                    raw_true, min_duration=1, background_id=Config.BACKGROUND_CLASS_ID
                )

                # Metric for this sample
                dist = levenshtein_distance(pred_seq, true_seq)

                all_preds.append(pred_seq)
                all_truths.append(true_seq)
                sample_errors.append(dist)

                # Collect features for analysis
                feat_seq_len.append(valid_len)
                feat_num_gestures.append(len(true_seq))

    # Compute Global Metric
    final_metric = compute_levenshtein_score(all_preds, all_truths)

    # Required Output Format
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("Performing failure analysis...")
    if len(sample_errors) > 1:
        # Check for variance to avoid division by zero in correlation
        if np.std(sample_errors) > 0 and np.std(feat_seq_len) > 0:
            corr_len, _ = pearsonr(sample_errors, feat_seq_len)
            print(f"Correlation (Error vs Seq Length): {corr_len:.4f}")
        else:
            print("Correlation (Error vs Seq Length): Undefined (variance=0)")

        if np.std(sample_errors) > 0 and np.std(feat_num_gestures) > 0:
            corr_count, _ = pearsonr(sample_errors, feat_num_gestures)
            print(f"Correlation (Error vs Num Gestures): {corr_count:.4f}")
        else:
            print("Correlation (Error vs Num Gestures): Undefined (variance=0)")

    # 5. Submission
    threshold = 0.061224489795918366

    if final_metric < threshold:
        print("Metric passed threshold. Generating submission...")

        test_dataset = GestureDataset(
            split="test", load_cached_data=True, transform=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_fn,
        )

        test_ids = []
        test_preds = []

        with torch.no_grad():
            for batch in test_loader:
                if batch is None:
                    continue

                skeleton = batch["skeleton"].to(device)
                audio = batch["audio"].to(device)
                lengths = batch["lengths"].to(device)
                ids = batch["sample_ids"]

                logits = model(skeleton, audio, lengths)
                preds = torch.argmax(logits, dim=2)

                for i in range(len(lengths)):
                    valid_len = lengths[i].item()
                    raw_pred = preds[i, :valid_len].cpu().numpy()

                    smoothed = apply_median_filter(raw_pred, kernel_size=5)
                    pred_seq = rle_decode(
                        smoothed,
                        min_duration=5,
                        background_id=Config.BACKGROUND_CLASS_ID,
                    )

                    test_ids.append(ids[i])
                    test_preds.append(pred_seq)

        # Format predictions
        lines = predictions_to_submission_format(test_ids, test_preds)

        # Save to file
        with open(Config.SUBMISSION_PATH, "w") as f:
            for line in lines:
                f.write(line + "\n")

        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(f"Metric {final_metric} >= {threshold}. Submission skipped.")


if __name__ == "__main__":
    main()
