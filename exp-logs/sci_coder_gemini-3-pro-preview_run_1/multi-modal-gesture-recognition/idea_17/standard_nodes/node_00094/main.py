import os
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Import from provided library modules
from library.config import (
    SEED,
    DEVICE,
    BATCH_SIZE,
    NUM_WORKERS,
    SUBMISSION_FILE,
    TEST_METADATA_PATH,
    MODEL_OUTPUT_CLASSES,
    BACKGROUND_CLASS_ID,
)
from library.utils import set_seed, decode_predictions, levenshtein_distance
from library.model import GCINet
from library.train import train_model, get_ground_truth_sequence
from library.data_loader import get_loaders


def main():
    # 1. Setup & Reproducibility
    set_seed(SEED)

    # 2. Train Model
    # We use 30 epochs for a fast baseline execution.
    # Given the small dataset size (~300 sequences), this allows for
    # sufficient convergence while keeping runtime minimal.
    print("Starting training pipeline...")
    best_model_path = train_model(num_epochs=30, batch_size=BATCH_SIZE)

    # 3. Validation & Failure Analysis
    print("Starting validation and failure analysis...")

    # Load the best model checkpoint
    model = GCINet().to(DEVICE)
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.eval()

    # Get data loaders
    _, val_loader, test_loader = get_loaders(
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    total_distance = 0
    total_gt_gestures = 0

    # Storage for failure analysis stats
    errors = []
    seq_lengths = []
    num_gestures = []

    with torch.no_grad():
        for batch in val_loader:
            skeleton = batch["skeleton"].to(DEVICE)
            audio = batch["audio"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            lengths = batch["lengths"].to(DEVICE)

            logits = model(skeleton, audio, lengths)

            # Iterate over each sample in the batch
            for i in range(logits.size(0)):
                length = lengths[i].item()

                # Extract valid portion of the sequence (ignoring padding)
                valid_logits = logits[i, :length, :]
                valid_labels = labels[i, :length]

                # Decode predictions and ground truth
                pred_seq = decode_predictions(valid_logits)
                gt_seq = get_ground_truth_sequence(valid_labels)

                # Compute Metric (Levenshtein Distance)
                dist = levenshtein_distance(pred_seq, gt_seq)

                total_distance += dist
                total_gt_gestures += len(gt_seq)

                # Collect stats for failure analysis
                errors.append(dist)
                seq_lengths.append(length)
                num_gestures.append(len(gt_seq))

    # Compute Final Metric
    if total_gt_gestures > 0:
        final_metric = total_distance / total_gt_gestures
    else:
        final_metric = 0.0

    # Print the required metric string
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation
    # We check if we have enough variance to compute correlation
    if len(errors) > 1:
        if np.std(errors) > 0 and np.std(seq_lengths) > 0:
            corr_len, _ = pearsonr(errors, seq_lengths)
            print(f"Correlation (Error vs Seq Length): {corr_len:.4f}")
        else:
            print("Correlation (Error vs Seq Length): Undefined (zero variance)")

        if np.std(errors) > 0 and np.std(num_gestures) > 0:
            corr_num, _ = pearsonr(errors, num_gestures)
            print(f"Correlation (Error vs Num Gestures): {corr_num:.4f}")
        else:
            print("Correlation (Error vs Num Gestures): Undefined (zero variance)")

    # 4. Submission
    THRESHOLD = 0.0765306122
    if final_metric < THRESHOLD:
        print(
            f"Metric ({final_metric}) is below threshold ({THRESHOLD}). Generating submission..."
        )

        # Load Test Metadata to ensure correct ID mapping (loader order matches metadata)
        test_df = pd.read_csv(TEST_METADATA_PATH)
        submission_lines = []
        sample_idx = 0

        with torch.no_grad():
            for batch in test_loader:
                skeleton = batch["skeleton"].to(DEVICE)
                audio = batch["audio"].to(DEVICE)
                lengths = batch["lengths"].to(DEVICE)

                logits = model(skeleton, audio, lengths)

                for i in range(logits.size(0)):
                    length = lengths[i].item()
                    valid_logits = logits[i, :length, :]

                    # Decode prediction
                    pred_seq = decode_predictions(valid_logits)

                    # Map to Sample ID
                    if sample_idx < len(test_df):
                        sample_id = test_df.iloc[sample_idx]["sample_id"]

                        # Format: SessionID,label1,label2...
                        labels_str = ",".join(map(str, pred_seq))
                        line = f"{sample_id},{labels_str}"
                        submission_lines.append(line)

                        sample_idx += 1

        # Write to file
        with open(SUBMISSION_FILE, "w") as f:
            for line in submission_lines:
                f.write(line + "\n")
        print(f"Submission saved to {SUBMISSION_FILE}")

    else:
        print(
            f"Metric ({final_metric}) >= threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
