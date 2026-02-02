import sys
import os
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import cv2

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, compute_levenshtein
from library.trainer import Trainer
from library.inference import generate_submission
from library.tokenizer import InChiTokenizer


def main():
    # =========================================================================
    # 1. Configuration for Fast Baseline
    # =========================================================================
    # Override Config defaults to fit within the 2-hour runtime limit
    # while ensuring enough data for a meaningful baseline.
    Config.EPOCHS = 2
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20000  # Train/Val on 20k samples each
    Config.BATCH_SIZE = 32  # EfficientNet/ResNet fit easily on A100
    Config.NUM_WORKERS = 4

    # Setup working directories
    Config.setup()

    print(
        f"Configuration Configured: Epochs={Config.EPOCHS}, Debug={Config.DEBUG}, Sample Size={Config.DEBUG_SAMPLE_SIZE}"
    )

    # =========================================================================
    # 2. Training
    # =========================================================================
    print("\nInitializing Trainer...")
    # Trainer internally handles data loading and model initialization
    trainer = Trainer(debug=Config.DEBUG)

    print("Starting Training...")
    trainer.fit()

    # =========================================================================
    # 3. Validation & Failure Analysis
    # =========================================================================
    print("\nStarting Validation and Failure Analysis...")

    # Ensure model is in eval mode
    trainer.model.eval()
    val_loader = trainer.val_loader
    tokenizer = trainer.tokenizer
    device = trainer.device

    predictions = []
    ground_truths = []
    lev_distances = []

    # Iterate through validation set to get detailed metrics
    with torch.no_grad():
        for i, (images, sequences, lengths) in enumerate(val_loader):
            images = images.to(device)

            # Predict using greedy decoding (equivalent to beam_width=1)
            # This is faster for validation than full beam search
            pred_indices = trainer.model.predict(images, max_len=Config.MAX_SEQ_LEN)

            # Decode batch
            for idx, idx_seq in enumerate(pred_indices):
                # Convert prediction indices to text
                pred_text = tokenizer.sequence_to_text(idx_seq)

                # Convert ground truth indices to text
                gt_seq = sequences[idx]
                gt_text = tokenizer.sequence_to_text(gt_seq)

                predictions.append(pred_text)
                ground_truths.append(gt_text)

                # Compute per-sample Levenshtein distance
                # We wrap in lists because compute_levenshtein expects lists
                d = compute_levenshtein([pred_text], [gt_text])
                lev_distances.append(d)

            if (i + 1) % 50 == 0:
                print(f"Validated {i + 1} batches...")

    # Compute Final Metric
    final_metric = np.mean(lev_distances)
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    print("\nPerforming Failure Analysis...")

    # 1. Target Lengths
    target_lengths = [len(t) for t in ground_truths]

    # 2. Image Dimensions
    # We need to read the image dimensions from disk corresponding to the validation set.
    # val_loader.dataset.df contains the metadata for the samples in the loader.
    val_df = val_loader.dataset.df

    # Safety check for length alignment
    if len(val_df) != len(lev_distances):
        print(
            f"Warning: Metadata length ({len(val_df)}) differs from prediction count ({len(lev_distances)}). Truncating to minimum."
        )
        min_len = min(len(val_df), len(lev_distances))
        val_df = val_df.iloc[:min_len]
        lev_distances = lev_distances[:min_len]
        target_lengths = target_lengths[:min_len]

    widths = []
    heights = []

    print("Reading image metadata for correlation analysis...")
    # Efficiently read image headers
    for _, row in val_df.iterrows():
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        try:
            # Read image to get dimensions (OpenCV is fast)
            img = cv2.imread(full_path)
            if img is not None:
                h, w, _ = img.shape
                widths.append(w)
                heights.append(h)
            else:
                widths.append(0)
                heights.append(0)
        except Exception:
            widths.append(0)
            heights.append(0)

    # Calculate Correlations
    # We check if error magnitude correlates with sequence length or image size
    if len(lev_distances) > 1:
        corr_len, _ = pearsonr(target_lengths, lev_distances)
        corr_width, _ = pearsonr(widths, lev_distances)
        corr_height, _ = pearsonr(heights, lev_distances)

        print(f"Correlation (Error vs Target Length): {corr_len:.4f}")
        print(f"Correlation (Error vs Image Width): {corr_width:.4f}")
        print(f"Correlation (Error vs Image Height): {corr_height:.4f}")
    else:
        print("Insufficient data for correlation analysis.")

    # =========================================================================
    # 4. Submission
    # =========================================================================
    threshold = 104.92673318379869

    if final_metric < threshold:
        print(
            f"\nValidation metric {final_metric} is better than threshold {threshold}."
        )
        print("Generating submission for the full test set...")

        # Generate submission
        # We use beam_width=1 (Greedy) to ensure inference fits within the remaining time.
        # debug=False ensures we process the *entire* test set as required.
        generate_submission(
            checkpoint_path=Config.CHECKPOINT_PATH,
            output_path=Config.SUBMISSION_PATH,
            batch_size=64,  # Increase batch size for faster inference on A100
            beam_width=1,  # Greedy decoding for speed
            debug=False,
        )
    else:
        print(f"\nValidation metric {final_metric} did not meet threshold {threshold}.")
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
