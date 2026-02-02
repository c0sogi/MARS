import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, compute_auc
from library.data_processing import prepare_data
from library.model import AVPFEModel
from library.training import train_one_epoch, generate_submission


def run_failure_analysis(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between error magnitude and continuous features.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_x_cont = []

    # Collect predictions, targets, and continuous features
    with torch.no_grad():
        for batch in val_loader:
            x_cont = batch["x_cont"].to(device)
            x_cat = batch["x_cat"].to(device)
            targets = batch["target"].to(device)

            logits = model(x_cont, x_cat)
            probs = torch.sigmoid(logits)
            avg_preds = torch.mean(probs, dim=1)

            all_preds.append(avg_preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_x_cont.append(x_cont.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_x_cont = np.concatenate(all_x_cont, axis=0)

    # Calculate Error Magnitude
    errors = np.abs(all_targets - all_preds)

    print("\n=== Failure Analysis ===")
    print("Correlation between Error Magnitude and Continuous Features:")

    # Calculate correlation for each continuous feature
    # We assume the order of columns matches the loader's x_cont
    # Based on data_processing, there are roughly 30 continuous features

    correlations = []
    for i in range(all_x_cont.shape[1]):
        # Pearson correlation
        if np.std(all_x_cont[:, i]) == 0:
            corr = 0.0
        else:
            corr, _ = pearsonr(errors, all_x_cont[:, i])
        correlations.append((i, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    for idx, corr in correlations[:10]:  # Print top 10
        print(f"Feature cont_{idx}: {corr:.6f}")

    return compute_auc(all_targets, all_preds)


def main():
    # 1. Setup and Config Override
    seed_everything(Config.SEED)

    # Override Config for Fast Baseline
    # Increasing to 30 epochs to allow full convergence for higher dropout streams
    Config.EPOCHS = 30

    print(f"Configuration:")
    print(f"  Device: {Config.DEVICE}")
    print(f"  Epochs: {Config.EPOCHS}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")

    # 2. Prepare Data
    print("Loading data...")
    train_loader, val_loader, test_loader, vocab_sizes = prepare_data(
        load_cached_data=True
    )

    # Determine input dimensions
    sample_batch = next(iter(train_loader))
    num_cont_features = sample_batch["x_cont"].shape[1]

    # 3. Initialize Model
    print("Initializing model...")
    model = AVPFEModel(vocab_sizes=vocab_sizes, num_cont_features=num_cont_features)
    model.to(Config.DEVICE)

    # 4. Optimizer and Scheduler
    optimizer = optim.Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_auc = 0.0

    print("Starting training loop...")
    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, Config.DEVICE, criterion
        )

        # Validate (using custom logic to ensure we track best model correctly)
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch in val_loader:
                x_cont = batch["x_cont"].to(Config.DEVICE)
                x_cat = batch["x_cat"].to(Config.DEVICE)
                targets = batch["target"].to(Config.DEVICE)

                logits = model(x_cont, x_cat)
                probs = torch.sigmoid(logits)
                avg_preds = torch.mean(probs, dim=1)

                val_preds.append(avg_preds.cpu())
                val_targets.append(targets.cpu())

        val_preds = torch.cat(val_preds).numpy()
        val_targets = torch.cat(val_targets).numpy()
        val_auc = compute_auc(val_targets, val_preds)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val AUC: {val_auc:.6f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), Config.MODEL_PATH)

    # 6. Final Evaluation and Failure Analysis
    print("\nLoading best model for final evaluation...")
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))

    final_auc = run_failure_analysis(model, val_loader, Config.DEVICE)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_auc}")

    # 7. Submission
    THRESHOLD = 0.9975746465492954
    if final_auc > THRESHOLD:
        print(f"Metric {final_auc} > {THRESHOLD}. Generating submission...")
        generate_submission()
    else:
        print(f"Metric {final_auc} <= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
