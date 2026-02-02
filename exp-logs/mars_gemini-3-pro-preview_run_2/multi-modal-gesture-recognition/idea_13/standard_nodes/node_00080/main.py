import os
import sys
import torch
import numpy as np
import pandas as pd

# Import functions and classes from the provided library files
from library.config import Config
from library.utils import set_seed, get_device, compute_levenshtein
from library.data_loader import get_dataloaders
from library.model import SBMD_CRCN
from library.losses import CombinedLoss
from library.train import train_one_epoch, validate, decode_sequence
from library.predict import generate_predictions


def analyze_failures(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Computes correlations between error rates and sequence properties.
    """
    print("\n--- Failure Analysis ---")
    model.eval()
    errors = []
    lengths = []
    num_targets = []

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            cls_targets = batch["cls_targets"].to(device)

            # Forward pass
            outputs = model(features, mask)

            # Use Stage 3 outputs for analysis
            logits = outputs["stage3_cls"]
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            # Move to CPU for processing
            preds_np = preds.cpu().numpy()
            targets_np = cls_targets.cpu().numpy()
            mask_np = mask.cpu().numpy()

            # Iterate through batch
            for i in range(preds_np.shape[0]):
                valid_len = int(mask_np[i].sum())

                # Extract valid sequence parts
                pred_raw = preds_np[i, :valid_len]
                target_raw = targets_np[i, :valid_len]

                # Decode to gesture IDs
                pred_seq = decode_sequence(pred_raw)
                true_seq = decode_sequence(target_raw)

                # Compute metric for this sample
                dist = compute_levenshtein(pred_seq, true_seq)

                errors.append(dist)
                lengths.append(valid_len)
                num_targets.append(len(true_seq))

    # Create DataFrame for statistical analysis
    df = pd.DataFrame({"error": errors, "length": lengths, "num_targets": num_targets})

    if len(df) > 1:
        # Compute correlations
        corr_len = df["error"].corr(df["length"])
        corr_num = df["error"].corr(df["num_targets"])

        print(f"Correlation between Error and Sequence Length: {corr_len:.4f}")
        print(f"Correlation between Error and Number of Gestures: {corr_num:.4f}")
        print(f"Mean Error per Sequence: {df['error'].mean():.4f}")
    else:
        print("Insufficient data for failure analysis.")


def main():
    # 1. Configuration
    set_seed(Config.SEED)
    device = get_device()
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading datasets...")
    # Load cached data to speed up initialization
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    print("Initializing model and optimizer...")
    model = SBMD_CRCN().to(device)
    criterion = CombinedLoss().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # 4. Check for existing best model
    THRESHOLD = 0.09294436906377204
    skip_training = False

    if os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Found existing checkpoint at {Config.BEST_MODEL_PATH}. Evaluating...")
        try:
            model.load_state_dict(
                torch.load(Config.BEST_MODEL_PATH, map_location=device)
            )
            _, val_score = validate(model, val_loader, criterion, device)
            print(f"Existing model validation score: {val_score:.6f}")

            if val_score < THRESHOLD:
                print(f"Model meets threshold ({THRESHOLD}). Skipping training.")
                skip_training = True
            else:
                print(
                    f"Model score {val_score:.6f} does not meet threshold ({THRESHOLD}). Retraining..."
                )
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Retraining...")

    # 5. Training Loop
    if not skip_training:
        print(f"Starting training for {Config.NUM_EPOCHS} epochs...")
        best_score = float("inf")

        for epoch in range(Config.NUM_EPOCHS):
            # Train
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )

            # Validate
            val_loss, val_score = validate(model, val_loader, criterion, device)

            print(
                f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Score: {val_score:.6f}"
            )

            # Save Best Model
            if val_score < best_score:
                best_score = val_score
                torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

        print("Training complete.")

    # 5. Final Evaluation
    print("Loading best model for final evaluation...")
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    else:
        print("Warning: Best model checkpoint not found.")

    # Compute final metric on validation set
    _, final_metric = validate(model, val_loader, criterion, device)

    # Print required metric format
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    analyze_failures(model, val_loader, device)

    # 7. Submission Logic
    THRESHOLD = 0.09294436906377204

    if final_metric < THRESHOLD:
        print(
            f"Metric {final_metric} meets threshold ({THRESHOLD}). Generating submission..."
        )
        generate_predictions(load_cached_data=True)
    else:
        print(
            f"Metric {final_metric} does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
