import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import from provided library files
from library.config import (
    WORKING_DIR,
    SEED,
    BATCH_SIZE,
    NUM_CLASSES,
    SUBMISSION_DIR,
    TEST_METADATA_PATH,
    VAL_METADATA_PATH,
)
from library.utils import (
    compute_levenshtein,
    post_process_output,
    rle_decode,
    calculate_levenshtein_accuracy,
)
from library.data_loader import MultimodalDataset, pad_collate, set_seed
from library.model import AGGRN
from library.train import train_model


def main():
    # 1. Setup
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Ensure submission dir exists
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # 2. Train Model
    # We use 30 epochs for a fast baseline as requested.
    # The dataset is small, so this should be very quick.
    print("Starting Training...")
    best_ler = train_model(load_cached_data=True, num_epochs=30, batch_size=BATCH_SIZE)

    # 3. Validation & Metric Calculation
    print("Loading best model for evaluation...")
    model = AGGRN().to(device)
    model_path = os.path.join(WORKING_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        print("Error: Best model checkpoint not found.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Create Val Loader manually for analysis
    val_ds = MultimodalDataset(split="val", load_cached_data=True)
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        collate_fn=pad_collate,
    )

    total_dist = 0
    total_ref_len = 0

    # Store per-sample stats for failure analysis
    sample_stats = []

    print("Running Inference on Validation Set...")
    with torch.no_grad():
        for batch in val_loader:
            skeleton, audio, labels, mask, lengths = batch

            skeleton = skeleton.to(device)
            audio = audio.to(device)
            # labels stays on cpu for decoding or move to device for loss?
            # In validation loop we need CPU for levenshtein usually

            logits = model(skeleton, audio, lengths)
            probs = torch.softmax(logits, dim=2)

            # Iterate batch
            for i in range(len(lengths)):
                length = lengths[i]
                seq_probs = probs[i, :length, :].cpu().numpy()
                seq_labels = labels[i, :length].numpy()  # Ground truth frame-wise

                # Decode
                pred_seq = post_process_output(
                    seq_probs, window_size=5, min_len=5, bg_class=0
                )
                target_seq = rle_decode(seq_labels, bg_class=0, min_len=1)

                dist = compute_levenshtein(pred_seq, target_seq)
                ref_len = len(target_seq)

                total_dist += dist
                total_ref_len += ref_len

                # Calculate normalized error for this sample (handle div by 0)
                sample_ler = (
                    dist / ref_len if ref_len > 0 else (1.0 if dist > 0 else 0.0)
                )

                sample_stats.append(
                    {
                        "num_frames": length.item(),
                        "num_gestures": ref_len,
                        "ler": sample_ler,
                        "dist": dist,
                    }
                )

    final_metric = total_dist / total_ref_len if total_ref_len > 0 else 0.0
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n--- Failure Analysis ---")
    df_stats = pd.DataFrame(sample_stats)

    if not df_stats.empty:
        # Correlation with Duration (Num Frames)
        if df_stats["num_frames"].std() > 0:
            corr_frames, _ = pearsonr(df_stats["ler"], df_stats["num_frames"])
            print(f"Correlation (Error vs NumFrames): {corr_frames:.4f}")
        else:
            print("Correlation (Error vs NumFrames): Undefined (constant variance)")

        # Correlation with Complexity (Num Gestures)
        if df_stats["num_gestures"].std() > 0:
            corr_gestures, _ = pearsonr(df_stats["ler"], df_stats["num_gestures"])
            print(f"Correlation (Error vs NumGestures): {corr_gestures:.4f}")
        else:
            print("Correlation (Error vs NumGestures): Undefined (constant variance)")

        print(f"Average Error Rate: {df_stats['ler'].mean():.4f}")
        print(f"Max Error Rate in a single sample: {df_stats['ler'].max():.4f}")

    # 5. Submission
    THRESHOLD = 0.0858843537414966
    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} < {THRESHOLD}. Generating submission...")

        test_ds = MultimodalDataset(split="test", load_cached_data=True)
        test_loader = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            collate_fn=pad_collate,
        )

        submission_lines = []

        with torch.no_grad():
            # We need to track sample IDs. The loader returns batches.
            # MultimodalDataset indices map to metadata rows.
            # But DataLoader shuffles? No, shuffle=False for test.
            # We can iterate the loader and the metadata index concurrently or retrieve ID from dataset.

            global_idx = 0

            for batch in test_loader:
                skeleton, audio, labels, mask, lengths = batch

                skeleton = skeleton.to(device)
                audio = audio.to(device)

                logits = model(skeleton, audio, lengths)
                probs = torch.softmax(logits, dim=2)

                for i in range(len(lengths)):
                    length = lengths[i]
                    seq_probs = probs[i, :length, :].cpu().numpy()

                    pred_seq = post_process_output(
                        seq_probs, window_size=5, min_len=5, bg_class=0
                    )

                    # Get Sample ID
                    # Since shuffle=False, we can access metadata by index
                    # Note: MultimodalDataset uses valid_indices.
                    # We need to get the sample_id corresponding to the current item.
                    # The dataset __getitem__ uses self.valid_indices[idx].

                    real_idx = test_ds.valid_indices[global_idx]
                    sample_id = test_ds.metadata.iloc[real_idx]["sample_id"]

                    # Format: SessionID,Label1,Label2...
                    line = f"{sample_id}," + ",".join(map(str, pred_seq))
                    submission_lines.append(line)

                    global_idx += 1

        submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        with open(submission_path, "w") as f:
            for line in submission_lines:
                f.write(line + "\n")

        print(f"Submission saved to {submission_path}")

    else:
        print(f"\nMetric {final_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
