import os
import torch
import numpy as np
import pandas as pd
import nltk
import library.config as config
from library.utils import set_seed
from library.data_loader import get_loaders
from library.model import DCSGCN
from library.loss import DCSGCNLoss
from library.train import train_one_epoch, validate, decode_predictions, decode_targets
from library.inference import run_inference


def analyze_failures(model, loader, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between error rate and sequence properties.
    """
    model.eval()
    errors = []
    seq_lengths_list = []
    num_gestures_list = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"]  # Keep on CPU for decoding
            lengths = batch["lengths"]

            # Forward pass
            outputs = model(features, mask)
            # Use Stage 3 outputs for final prediction
            cls_probs, _ = outputs["stage3"]
            cls_probs_np = cls_probs.cpu().numpy()

            # Decode predictions and targets
            preds = decode_predictions(cls_probs_np, lengths)
            targets = decode_targets(labels, lengths)

            # Calculate per-sample metrics
            for p, t, l in zip(preds, targets, lengths):
                dist = nltk.edit_distance(p, t)
                t_len = len(t)

                # Calculate error rate per sample
                # If target length is 0, use raw distance to avoid division by zero
                if t_len > 0:
                    err = dist / t_len
                else:
                    err = float(dist)

                errors.append(err)
                seq_lengths_list.append(l.item())
                num_gestures_list.append(t_len)

    # Compute Correlations
    if errors:
        df = pd.DataFrame(
            {
                "error": errors,
                "seq_length": seq_lengths_list,
                "num_gestures": num_gestures_list,
            }
        )

        # Pearson correlation
        corr_len = df["error"].corr(df["seq_length"])
        corr_num = df["error"].corr(df["num_gestures"])

        print(f"Correlation (Error vs Sequence Length): {corr_len:.4f}")
        print(f"Correlation (Error vs Num Gestures): {corr_num:.4f}")
    else:
        print("No validation samples found for analysis.")


def main():
    # 1. Setup
    set_seed(config.SEED)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Fast Baseline Configuration ---
    # Override config to ensure quick execution within time limits
    config.HYPERPARAMS["num_epochs"] = 30
    config.HYPERPARAMS["patience"] = 8

    # 2. Data Loading
    # load_cached_data is True by default in GestureDataset
    print("Loading DataLoaders...")
    train_loader, val_loader, test_loader = get_loaders()

    # 3. Model & Loss Initialization
    model = DCSGCN().to(device)
    criterion = DCSGCNLoss().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.HYPERPARAMS["lr"],
        weight_decay=config.HYPERPARAMS["weight_decay"],
    )

    # 4. Training Loop
    print(f"Starting training for {config.HYPERPARAMS['num_epochs']} epochs...")

    best_val_metric = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")

    for epoch in range(config.HYPERPARAMS["num_epochs"]):
        # Train one epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_lev = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{config.HYPERPARAMS['num_epochs']} "
            f"- Train Loss: {train_loss:.4f} "
            f"- Val Loss: {val_loss:.4f} "
            f"- Val Levenshtein: {val_lev:.4f}"
        )

        # Checkpointing based on Levenshtein Metric (Primary Goal)
        if val_lev < best_val_metric:
            best_val_metric = val_lev
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= config.HYPERPARAMS["patience"]:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # 5. Final Evaluation
    print("Training complete. Loading best model...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    else:
        print("Warning: No best model found. Using current weights.")

    # Compute final metric on full validation set
    _, final_metric = validate(model, val_loader, criterion, device)

    # Print exactly as requested
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("Running Failure Analysis...")
    analyze_failures(model, val_loader, device)

    # 7. Submission Logic
    threshold = 0.06789606035205364
    if final_metric < threshold:
        print(f"Metric {final_metric} < {threshold}. Generating submission...")
        submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
        run_inference(
            checkpoint_path=best_model_path, output_path=submission_path, device=device
        )
    else:
        print(f"Metric {final_metric} >= {threshold}. Submission skipped.")


if __name__ == "__main__":
    main()
