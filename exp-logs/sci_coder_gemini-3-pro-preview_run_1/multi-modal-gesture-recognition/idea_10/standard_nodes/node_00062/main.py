import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.stats import pearsonr

# Import provided library components
from library.config import (
    NUM_CLASSES,
    BACKGROUND_CLASS_ID,
    BACKGROUND_WEIGHT,
    LABEL_SMOOTHING,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    BEST_MODEL_PATH,
    SEED,
    SUBMISSION_FILE_PATH,
    TEST_METADATA_PATH,
    VAL_METADATA_PATH,
)
from library.utils import set_seed, levenshtein_distance, rle_decode, median_filter
from library.data_loader import get_dataloaders
from library.model import DGR_RN
from library.train import train_one_epoch, validate


def analyze_failures(model, loader, device, metadata_path):
    """
    Runs inference on the loader, computes per-sample metrics,
    and performs failure analysis.
    """
    model.eval()

    # Load metadata to map indices to IDs and get ground truth if needed
    df = pd.read_csv(metadata_path)

    errors = []
    lengths = []

    # We iterate sequentially; loader must be shuffle=False
    sample_idx = 0

    with torch.no_grad():
        for batch in loader:
            skeleton = batch["skeleton"].to(device)
            audio = batch["audio"].to(device)
            labels = batch["labels"].to(device)
            batch_lengths = batch["lengths"]

            # Forward
            logits = model(skeleton, audio)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()
            probs_np = probs.cpu().numpy()

            for i in range(len(labels)):
                length = batch_lengths[i].item()

                # Extract valid sequence
                curr_probs = probs_np[i, :length, :]
                curr_labels = labels_np[i, :length]

                # Apply smoothing
                smoothed_probs = median_filter(curr_probs, window_size=5)
                smoothed_preds = np.argmax(smoothed_probs, axis=-1)

                # Decode
                hyp_seq = rle_decode(smoothed_preds)
                ref_seq = rle_decode(curr_labels)

                # Compute Distance
                dist = levenshtein_distance(hyp_seq, ref_seq)

                # Store stats
                errors.append(dist)
                lengths.append(length)

                sample_idx += 1

    # Correlation Analysis
    if len(errors) > 1:
        corr, _ = pearsonr(errors, lengths)
        print(f"Correlation between Error and Sequence Length: {corr:.4f}")
    else:
        print("Not enough samples for correlation analysis.")


def generate_submission(model, loader, device, metadata_path, output_path):
    """
    Generates predictions for the test set and saves to CSV.
    """
    model.eval()
    df = pd.read_csv(metadata_path)
    results = []

    # Ensure sequential matching
    sample_idx = 0

    with torch.no_grad():
        for batch in loader:
            skeleton = batch["skeleton"].to(device)
            audio = batch["audio"].to(device)
            batch_lengths = batch["lengths"]

            logits = model(skeleton, audio)
            probs = torch.softmax(logits, dim=-1)

            probs_np = probs.cpu().numpy()

            for i in range(len(logits)):
                # Get Sample ID from metadata
                if sample_idx >= len(df):
                    break
                sample_id = df.iloc[sample_idx]["sample_id"]

                length = batch_lengths[i].item()
                curr_probs = probs_np[i, :length, :]

                # Smoothing and Decoding
                smoothed_probs = median_filter(curr_probs, window_size=5)
                smoothed_preds = np.argmax(smoothed_probs, axis=-1)

                hyp_seq = rle_decode(smoothed_preds)

                # Format string: "ID,label1,label2,..."
                pred_str = ",".join(map(str, hyp_seq))
                results.append(f"{sample_id},{pred_str}")

                sample_idx += 1

    # Write to file
    with open(output_path, "w") as f:
        for line in results:
            f.write(line + "\n")

    print(f"Submission saved to {output_path}")


def main():
    # 1. Setup
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Data Loading
    # Using load_cached_data=True implicitly via get_dataloaders defaults/implementation
    train_loader, val_loader, test_loader = get_dataloaders()

    # 3. Model Initialization
    model = DGR_RN().to(device)

    # 4. Training Setup
    weights = torch.ones(NUM_CLASSES).to(device)
    weights[BACKGROUND_CLASS_ID] = BACKGROUND_WEIGHT
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)

    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # 5. Training Loop
    best_ler = float("inf")

    print("Starting training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_ler = validate(model, val_loader, device)
        scheduler.step()

        # Save best model
        if val_ler < best_ler:
            best_ler = val_ler
            torch.save(model.state_dict(), BEST_MODEL_PATH)

    print(f"Training finished. Best LER: {best_ler:.6f}")

    # 6. Final Evaluation & Failure Analysis
    # Load best model
    model.load_state_dict(torch.load(BEST_MODEL_PATH))

    # Re-calculate metric on full validation set to ensure accuracy and print required format
    final_val_ler = validate(model, val_loader, device)
    print(f"Final Validation Metric: {final_val_ler}")

    print("Performing Failure Analysis on Validation Set...")
    analyze_failures(model, val_loader, device, VAL_METADATA_PATH)

    # 7. Submission
    THRESHOLD = 0.0824829931972789
    if final_val_ler < THRESHOLD:
        print(
            f"Validation metric ({final_val_ler}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(
            model, test_loader, device, TEST_METADATA_PATH, SUBMISSION_FILE_PATH
        )
    else:
        print(
            f"Validation metric ({final_val_ler}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
