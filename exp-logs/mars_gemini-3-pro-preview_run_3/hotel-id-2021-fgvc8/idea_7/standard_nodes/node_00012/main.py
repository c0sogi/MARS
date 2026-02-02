import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import library components
from library.config import Config, seed_everything
from library.dataset import get_dataloaders
from library.model import HotelIdModel
from library.engine import (
    train_fn,
    validate_fn,
    generate_submission,
    inference_fn,
    get_nearest_neighbors,
)
from library.utils import save_checkpoint, apk


def run():
    # 1. Setup
    seed_everything(Config.seed)
    device = Config.device
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading data...")
    # Load dataloaders with caching enabled for speed
    train_loader, val_loader, test_loader, unique_ids = get_dataloaders(
        load_cached_data=True
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = HotelIdModel()
    model.to(device)

    # 4. Training Configuration
    # ArcFace outputs logits, so CrossEntropyLoss is appropriate
    criterion = nn.CrossEntropyLoss()

    # Optimizer: AdamW
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )

    # Scheduler: Cosine Annealing
    # Steps once per epoch
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.epochs, eta_min=Config.min_lr)

    # 5. Training Loop
    best_map = 0.0

    print(f"Starting training for {Config.epochs} epochs...")
    for epoch in range(Config.epochs):
        # Run training epoch
        train_loss, train_acc = train_fn(
            train_loader, model, criterion, optimizer, device, epoch
        )

        # Step scheduler
        scheduler.step()

        # Run validation
        val_map = validate_fn(val_loader, model, device, unique_ids)

        # Save best model
        is_best = val_map > best_map
        if is_best:
            best_map = val_map
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_map": best_map,
                    "optimizer": optimizer.state_dict(),
                },
                is_best=True,
            )
            print(f"New best model saved with MAP@5: {best_map:.6f}")

    # 6. Final Evaluation & Failure Analysis
    print("\nLoading best model for final evaluation...")
    checkpoint_path = os.path.join(Config.output_dir, "best_model.pth")
    # Load on CPU first to avoid memory spikes, then move to device
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    # Compute final metric on the full validation set
    final_metric = validate_fn(val_loader, model, device, unique_ids)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\nPerforming failure analysis...")
    # Get predictions (features) and ground truth labels from validation set
    features, labels = inference_fn(val_loader, model, device)
    preds_indices = get_nearest_neighbors(features, model, device, k=5)

    # Convert tensors to lists for calculation
    actual = [[l.item()] for l in labels]
    predicted = preds_indices.tolist()

    # Calculate AP@5 for each individual sample
    ap_scores = [apk(a, p, k=5) for a, p in zip(actual, predicted)]

    # Load training metadata to get class frequencies (to check for long-tail issues)
    train_df = pd.read_csv(Config.train_metadata_path)
    val_df = val_loader.dataset.df

    # Calculate training class counts
    class_counts = train_df["hotel_id"].value_counts().to_dict()

    # Map these counts to the validation samples
    # This creates an array aligned with 'ap_scores' representing how many training examples existed for that target
    val_counts = val_df["hotel_id"].map(class_counts).fillna(0).values

    # Calculate correlation between Performance (AP Score) and Class Frequency
    # A positive correlation indicates the model performs better on frequent classes (common in long-tail data)
    correlation = np.corrcoef(ap_scores, val_counts)[0, 1]
    print(f"Correlation between Error (Performance) and Class Frequency: {correlation}")

    # 7. Submission
    submission_threshold = 0.5589516758918762
    if final_metric > submission_threshold:
        generate_submission(test_loader, model, device, unique_ids)
    else:
        print(
            f"Validation metric {final_metric} did not exceed threshold {submission_threshold}. Skipping submission."
        )


if __name__ == "__main__":
    run()
