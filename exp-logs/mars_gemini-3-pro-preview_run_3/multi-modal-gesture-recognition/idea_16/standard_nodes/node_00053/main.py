import os
import shutil
import warnings
import torch
import numpy as np
import torch.nn.functional as F
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library.utils import (
    set_seed,
    setup_logger,
    rle_encode_predictions,
    compute_levenshtein_distance,
)
from library.train import run_training
from library.inference import run_inference
from library.model import RDKRN
from library.data_loader import get_dataloaders


def main():
    # 1. Setup
    # Suppress warnings for clean output
    warnings.filterwarnings("ignore")

    # Set seeds for reproducibility
    set_seed(Config.SEED)

    # Ensure submission directory exists
    if not os.path.exists("./submission"):
        os.makedirs("./submission")

    # 2. Training
    # We use the full dataset (limit_samples=None) and default epochs (60)
    # as the dataset is small and we have GPU acceleration.
    print("Starting training pipeline...")
    run_training(limit_samples=None, num_epochs=Config.NUM_EPOCHS)

    # 3. Validation & Metric Calculation
    print("Starting validation and failure analysis...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the best model saved during training
    model = RDKRN().to(device)
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Error: Model file not found at {Config.MODEL_SAVE_PATH}")
        return

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # Get Validation Loader
    _, val_loader, _ = get_dataloaders(batch_size=Config.BATCH_SIZE, limit_samples=None)
    val_dataset = val_loader.dataset

    # Prepare buffers for full sequence reconstruction (Temporal Ensembling)
    # This mirrors the inference logic to ensure the metric is accurate.
    buffers = {}
    for _, length, sample_id in val_dataset.seq_info:
        buffers[sample_id] = {
            "probs": np.zeros((length, Config.NUM_CLASSES), dtype=np.float32),
            "counts": np.zeros((length, 1), dtype=np.float32),
            "targets": np.zeros((length,), dtype=np.int32),
        }

    # Run Inference on Validation Set
    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            targets = batch["targets"].cpu().numpy()
            sample_ids = batch["sample_id"]
            frame_starts = batch["frame_start"]

            # Forward pass
            outputs = model(features)
            # Use Stage 3 outputs
            stage3_logits = outputs[2]
            probs = F.softmax(stage3_logits, dim=2).cpu().numpy()

            # Aggregate into buffers
            for i, sid in enumerate(sample_ids):
                if sid not in buffers:
                    continue

                start = frame_starts[i]
                p = probs[i]
                t = targets[i]

                # Determine valid range
                buffer_len = buffers[sid]["probs"].shape[0]
                window_len = p.shape[0]
                end = min(start + window_len, buffer_len)
                valid_len = end - start

                if valid_len > 0:
                    buffers[sid]["probs"][start:end] += p[:valid_len]
                    buffers[sid]["counts"][start:end] += 1.0
                    # Store targets (overwrite is fine as they are consistent)
                    buffers[sid]["targets"][start:end] = t[:valid_len]

    # Decode sequences and compute metrics
    total_distance = 0
    total_gestures = 0

    # For Failure Analysis
    sample_errors = []
    sample_lengths = []
    sample_num_gestures = []

    for sid in buffers:
        data = buffers[sid]

        # Average probabilities
        counts = data["counts"]
        counts[counts == 0] = 1.0  # Prevent division by zero
        avg_probs = data["probs"] / counts

        # Decode
        pred_labels = np.argmax(avg_probs, axis=1)
        target_labels = data["targets"]

        # RLE Encode
        pred_seq = rle_encode_predictions(pred_labels, Config.BACKGROUND_CLASS_ID)
        target_seq = rle_encode_predictions(target_labels, Config.BACKGROUND_CLASS_ID)

        # Compute Distance
        dist = compute_levenshtein_distance(pred_seq, target_seq)

        total_distance += dist
        total_gestures += len(target_seq)

        # Collect stats
        sample_errors.append(dist)
        sample_lengths.append(len(target_labels))
        sample_num_gestures.append(len(target_seq))

    # Compute Final Metric
    final_metric = total_distance / total_gestures if total_gestures > 0 else 1.0

    # Print required output
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    if len(sample_errors) > 1:
        # Correlation with Sequence Length
        corr_len, _ = pearsonr(sample_errors, sample_lengths)
        # Correlation with Number of Gestures
        corr_num, _ = pearsonr(sample_errors, sample_num_gestures)

        print("Failure Analysis on Validation Set:")
        print(f"Correlation (Error vs Sequence Length): {corr_len:.4f}")
        print(f"Correlation (Error vs Num Gestures): {corr_num:.4f}")

    # 5. Submission Generation
    THRESHOLD = 0.2251
    if final_metric < THRESHOLD:
        print(
            f"Metric {final_metric:.4f} is below threshold {THRESHOLD}. Generating submission..."
        )

        # Run inference on test set
        run_inference(limit_samples=None)

        # Copy submission to required path
        src_path = Config.SUBMISSION_PATH
        dst_path = "./submission/submission.csv"

        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
            print(f"Submission file successfully saved to {dst_path}")
        else:
            print(f"Error: Source submission file not found at {src_path}")
    else:
        print(
            f"Metric {final_metric:.4f} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
