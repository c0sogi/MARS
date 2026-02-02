import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.utils import seed_everything, compute_levenshtein
from library.tokenizer import Tokenizer
from library.dataset import get_val_dataloader
from library.model import ShowAttendTell
from library.train import fit
from library.predict import generate_predictions


def main():
    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    print("Initializing Fast Baseline Run...")

    # Override Config for a fast baseline execution within 2 hours
    Config.EPOCHS = 1
    Config.DEBUG_SAMPLE_SIZE = 50000  # Train on 50k samples (~3% of data)

    # Enable cuDNN benchmark for speed (fixed input size)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Ensure reproducibility
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # ---------------------------------------------------------
    # 2. Training
    # ---------------------------------------------------------
    print("\n" + "=" * 40)
    print("STEP 1: Training Model")
    print("=" * 40)

    # Run training loop
    # debug=True ensures we use the subset defined by DEBUG_SAMPLE_SIZE
    fit(debug=True, load_cached_data=True)

    # ---------------------------------------------------------
    # 3. Validation & Failure Analysis
    # ---------------------------------------------------------
    print("\n" + "=" * 40)
    print("STEP 2: Validation & Failure Analysis")
    print("=" * 40)

    # Load Tokenizer
    tokenizer = Tokenizer(load_cached_data=True)
    vocab_size = len(tokenizer)

    # Load Best Model
    print(f"Loading best model from {Config.MODEL_SAVE_PATH}...")
    model = ShowAttendTell(vocab_size=vocab_size).to(device)
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError("Model checkpoint not found. Training may have failed.")

    checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Load Full Validation Dataset
    # debug=False ensures we use the entire validation set (~380k samples)
    print("Loading full validation dataset...")
    val_loader = get_val_dataloader(
        tokenizer, batch_size=Config.BATCH_SIZE, debug=False, load_cached_data=True
    )

    lev_distances = []
    target_lengths = []

    print("Starting validation inference...")
    start_val = time.time()

    with torch.no_grad():
        for i, (images, captions, lengths) in enumerate(val_loader):
            images = images.to(device)

            # Forward pass in inference mode (greedy decoding)
            outputs = model(images, captions=None)

            # Get predictions
            predicted_indices = torch.argmax(outputs, dim=2)

            # Process batch for metrics
            # Move to CPU once to minimize overhead
            pred_batch = predicted_indices.cpu().tolist()
            tgt_batch = captions.cpu().tolist()

            for idx in range(len(pred_batch)):
                # Decode text
                pred_text = tokenizer.sequence_to_text(pred_batch[idx])
                tgt_text = tokenizer.sequence_to_text(tgt_batch[idx])

                # Compute Metric
                dist = compute_levenshtein([pred_text], [tgt_text])
                lev_distances.append(dist)

                # Collect feature for failure analysis (Target String Length)
                target_lengths.append(len(tgt_text))

            if i % 100 == 0:
                print(f"Validated batch {i}/{len(val_loader)}")

    val_time = time.time() - start_val
    print(f"Validation completed in {val_time:.2f} seconds.")

    # Final Metric
    final_metric = np.mean(lev_distances)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\n[Failure Analysis]")
    if len(lev_distances) > 1:
        correlation, p_val = pearsonr(lev_distances, target_lengths)
        print(f"Correlation (Error vs Target Length): {correlation:.4f}")
        print(f"P-value: {p_val:.4e}")
        if abs(correlation) > 0.3:
            print(
                "Observation: Significant correlation detected. Model struggles with longer/complex sequences."
            )
        else:
            print(
                "Observation: Low correlation. Error is distributed across sequence lengths."
            )
    else:
        print("Not enough samples for correlation analysis.")

    # ---------------------------------------------------------
    # 4. Submission
    # ---------------------------------------------------------
    print("\n" + "=" * 40)
    print("STEP 3: Generating Submission")
    print("=" * 40)

    # Generate predictions for the full test set
    # debug=False ensures full test set
    generate_predictions(debug=False, load_cached_data=True)

    print("\nPipeline execution complete.")


if __name__ == "__main__":
    main()
