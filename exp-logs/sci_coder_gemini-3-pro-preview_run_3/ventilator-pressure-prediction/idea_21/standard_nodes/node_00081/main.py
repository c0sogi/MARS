import sys
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

# Ensure library is in path
sys.path.append(os.getcwd())

from library.config import Config
from library.model import RDHNet
from library.data import load_and_preprocess
from library.utils import seed_everything, compute_mae
from library.train import train_epoch, validate_epoch, generate_submission


def main():
    # 1. Configuration
    # Use Config epochs (100) for full convergence (Cite solution_lesson_node_00039)
    Config.setup()
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print("Configuration:")
    print(f"  Device: {device}")
    print(f"  Epochs: {Config.EPOCHS}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  CNN Filters: {Config.CNN_FILTERS}")

    # 2. Data Loading
    print("\nLoading Data...")
    train_ds, val_ds, test_ds = load_and_preprocess(load_cached_data=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("\nInitializing Model...")
    model = RDHNet().to(device)

    # 4. Optimization
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    num_training_steps = len(train_loader) * Config.EPOCHS
    num_warmup_steps = len(train_loader) * Config.WARMUP_EPOCHS

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=Config.COSINE_CYCLES,
    )

    # 5. Training Loop
    print("\nStarting Training...")
    best_val_mae = float("inf")

    for epoch in range(1, Config.EPOCHS + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, Config.MAX_GRAD_NORM
        )
        val_loss, val_mae = validate_epoch(model, val_loader, device)

        print(
            f"Epoch {epoch}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val MAE: {val_mae:.6f}"
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  -> New Best Model Saved! (MAE: {best_val_mae:.6f})")

    # 6. Final Evaluation & Failure Analysis
    print("\nLoading Best Model for Evaluation...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    print("Computing Final Metrics and Failure Analysis...")
    all_preds = []
    all_targets = []
    all_inputs = []
    all_u_out = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            u_out = batch["u_out"].to(device)

            preds = model(inputs)

            # Move to CPU to save GPU memory during collection
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            all_inputs.append(inputs.cpu())
            all_u_out.append(u_out.cpu())

    # Concatenate
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_inputs = torch.cat(all_inputs, dim=0)
    all_u_out = torch.cat(all_u_out, dim=0)

    # Final Metric
    final_mae = compute_mae(all_preds, all_targets, all_u_out)
    print(f"Final Validation Metric: {final_mae}")

    # Failure Analysis
    # Filter for inspiratory phase (u_out == 0)
    mask = (all_u_out == 0).bool()

    insp_inputs = all_inputs[mask].numpy()
    insp_preds = all_preds[mask].numpy()
    insp_targets = all_targets[mask].numpy()

    errors = np.abs(insp_preds - insp_targets)

    # Create DataFrame
    analysis_df = pd.DataFrame(insp_inputs, columns=Config.INPUT_FEATURES)
    analysis_df["error_magnitude"] = errors

    # Correlation
    correlations = (
        analysis_df.corr()["error_magnitude"]
        .drop("error_magnitude")
        .sort_values(ascending=False)
    )
    print("\nCorrelation between Error Magnitude and Input Features (Validation Set):")
    print(correlations)

    # 7. Submission
    THRESHOLD = 0.16391726930343686
    if final_mae < THRESHOLD:
        print(
            f"\nMetric ({final_mae}) < Threshold ({THRESHOLD}). Generating Submission..."
        )
        generate_submission(
            model, test_loader, device, Config.SUBMISSION_PATH, Config.CACHE_DIR
        )
    else:
        print(f"\nMetric ({final_mae}) >= Threshold ({THRESHOLD}). Submission Skipped.")


if __name__ == "__main__":
    main()
