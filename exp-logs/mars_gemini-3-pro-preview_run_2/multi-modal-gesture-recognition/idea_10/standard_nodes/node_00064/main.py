import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, decode_predictions, compute_levenshtein_distance
from library.data_loader import get_dataloaders
from library.model import GMD_CRCN
from library.train import train_model


def main():
    # =========================================================================
    # 1. Configuration Override for Fast Baseline
    # =========================================================================
    # We override default config values to ensure the script completes within the
    # 2-hour time limit while still providing a meaningful baseline.
    Config.NUM_EPOCHS = 60
    Config.PATIENCE = 15

    # Invalidate stale cache to force data regeneration (Cite debug_lesson_4)
    if os.path.exists(Config.CACHE_DIR):
        import shutil

        print(f"Removing stale cache at {Config.CACHE_DIR}...")
        shutil.rmtree(Config.CACHE_DIR)

    # Initialize directories (creates CHECKPOINT_DIR, CACHE_DIR, etc.)
    Config.setup()

    # Ensure deterministic behavior
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # =========================================================================
    # 2. Training Phase
    # =========================================================================
    print("\n=== Starting Training Phase ===")
    # train_model initializes loaders, model, optimizer, and runs the training loop.
    # It returns the best Levenshtein Error Rate achieved on the validation set.
    best_val_score = train_model(epochs=Config.NUM_EPOCHS, patience=Config.PATIENCE)

    # REQUIRED: Print Final Validation Metric in the specific format
    print(f"Final Validation Metric: {best_val_score}")

    # =========================================================================
    # 3. Failure Analysis Phase
    # =========================================================================
    print("\n=== Starting Failure Analysis Phase ===")

    # Initialize model and load the best checkpoint
    model = GMD_CRCN().to(device)
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Loaded best model checkpoint for analysis.")
    else:
        print("Warning: Checkpoint not found. Analysis will use current model state.")

    model.eval()

    # Get validation loader (using default batch size for efficiency)
    _, val_loader, _ = get_dataloaders(batch_size=Config.BATCH_SIZE)

    val_errors = []
    val_lengths = []

    print("Computing error statistics on validation set...")
    with torch.no_grad():
        for batch in val_loader:
            pos = batch["pos"].to(device)
            vel = batch["vel"].to(device)
            audio = batch["audio"].to(device)
            labels = batch["labels"].to(device)
            lengths = batch["lengths"].to(device)

            # Forward pass
            predictions = model(pos, vel, audio, lengths)

            # Use Stage 3 output for final predictions
            logits = predictions["stage3"]
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            # Convert to CPU for metric calculation
            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()
            lengths_np = lengths.cpu().numpy()

            # Iterate through batch
            for i in range(len(lengths_np)):
                l = lengths_np[i]
                # Slice sequence to valid length
                p_seq = preds_np[i, :l]
                t_seq = labels_np[i, :l]

                # Decode: Collapse repeats and remove background (class 0)
                dec_p = decode_predictions(
                    p_seq, collapse_repeats=True, remove_background=True
                )
                dec_t = decode_predictions(
                    t_seq, collapse_repeats=True, remove_background=True
                )

                # Compute Levenshtein distance
                dist = compute_levenshtein_distance(dec_p, dec_t)

                val_errors.append(dist)
                val_lengths.append(l)

    # Compute Correlation
    if len(val_errors) > 1:
        corr, _ = pearsonr(val_errors, val_lengths)
        print(f"Correlation between Error Magnitude and Sequence Length: {corr:.6f}")
        print(f"Average Levenshtein Error: {np.mean(val_errors):.4f}")
        print(f"Max Error: {np.max(val_errors)}")
    else:
        print("Insufficient data for correlation analysis.")

    # =========================================================================
    # 4. Submission Generation Phase
    # =========================================================================
    THRESHOLD = 0.10854816824966079

    if best_val_score < THRESHOLD:
        print(
            f"\n=== Generating Submission (Score {best_val_score:.6f} < {THRESHOLD}) ==="
        )

        # IMPORTANT: Use Batch Size = 1 for inference.
        # The collate_fn sorts batches by length, which shuffles the order relative to the dataset.
        # With batch_size=1, sorting is a no-op, preserving the alignment with sample IDs.
        _, _, test_loader_seq = get_dataloaders(batch_size=1)

        # Load Test Metadata to retrieve Sample IDs
        test_df = pd.read_csv(Config.TEST_CSV)

        submission_lines = []

        with torch.no_grad():
            # Enumerate to align with DataFrame index
            for idx, batch in enumerate(test_loader_seq):
                pos = batch["pos"].to(device)
                vel = batch["vel"].to(device)
                audio = batch["audio"].to(device)
                lengths = batch["lengths"].to(device)

                # Forward pass
                predictions = model(pos, vel, audio, lengths)

                # Decode Stage 3
                logits = predictions["stage3"]
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                # Extract prediction (batch size is 1)
                l = lengths[0].item()
                p_seq = preds[0, :l].cpu().numpy()

                # Decode gestures
                decoded_gestures = decode_predictions(
                    p_seq, collapse_repeats=True, remove_background=True
                )

                # Get corresponding Sample ID from metadata
                sample_id = test_df.iloc[idx]["sample_id"]

                # Format: SessionID,gesture1,gesture2,...
                gesture_str = ",".join(map(str, decoded_gestures))
                line = f"{sample_id},{gesture_str}"
                submission_lines.append(line)

        # Save submission file
        sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        with open(sub_path, "w") as f:
            for line in submission_lines:
                f.write(line + "\n")

        print(f"Submission saved to {sub_path}")
        print(f"Generated predictions for {len(submission_lines)} sequences.")

    else:
        print(
            f"\n=== Skipping Submission (Score {best_val_score:.6f} >= {THRESHOLD}) ==="
        )


if __name__ == "__main__":
    main()
