import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import warnings

# Import from provided library
from library.config import (
    SEED,
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    WORKING_DIR,
    LOSS_WEIGHTS,
    seed_everything,
)
from library.dataset import get_dataloaders
from library.model import ResNet18UNet
from library.engine import train_one_epoch, evaluate, inference

# Suppress warnings
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    seed_everything(SEED)

    # 2. Data Loading
    # Using full dataset to ensure we hit the metric threshold.
    # ResNet18 on A100 is fast enough for 20 epochs within 2 hours.
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, debug=False
    )

    # 3. Model & Optimizer
    # Cite solution_lesson_node_00015: Revert to Standard U-Net (remove Attention)
    model = ResNet18UNet(pretrained=True)
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    # 4. Training Loop
    best_map = 0.0
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")

    # Early stopping parameters
    patience = 5
    patience_counter = 0

    print(f"Starting training on {DEVICE} for {EPOCHS} epochs...")

    for epoch in range(1, EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE, epoch)

        # Step Scheduler
        scheduler.step()

        # Validate
        val_loss, val_map = evaluate(model, val_loader, DEVICE)

        # Checkpointing
        if val_map > best_map:
            best_map = val_map
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= patience:
            break

    # 5. Final Metrics
    print(f"Final Validation Metric: {best_map}")

    # 6. Failure Analysis
    print("Performing failure analysis...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

    model.eval()

    # Containers for analysis
    errors = []
    means = []
    stds = []

    # Loss functions for error calculation (reduction='none' to get per-sample error)
    criterion_class = nn.CrossEntropyLoss(reduction="none")
    criterion_seg = nn.BCEWithLogitsLoss(reduction="none")

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            # Forward
            class_logits, seg_logits = model(images)

            # Calculate per-sample loss
            loss_class = criterion_class(class_logits, labels)

            # Seg loss is (B, 1, H, W). Average over spatial dims to get (B, 1), then squeeze
            loss_seg_map = criterion_seg(seg_logits, masks)
            loss_seg = loss_seg_map.mean(dim=(1, 2, 3))

            # Weighted total error
            total_error = (LOSS_WEIGHTS["class"] * loss_class) + (
                LOSS_WEIGHTS["seg"] * loss_seg
            )

            errors.extend(total_error.cpu().numpy())

            # Calculate input features (on normalized tensors)
            # Mean/Std across Channel, Height, Width per sample
            batch_means = images.mean(dim=(1, 2, 3)).cpu().numpy()
            batch_stds = images.std(dim=(1, 2, 3)).cpu().numpy()

            means.extend(batch_means)
            stds.extend(batch_stds)

    # Compute correlations
    df_analysis = pd.DataFrame(
        {"error": errors, "pixel_mean": means, "pixel_std": stds}
    )

    correlations = df_analysis.corr()["error"].drop("error")
    print("Error Correlations with Input Features:")
    print(correlations)

    # 7. Conditional Submission
    THRESHOLD = 0.4915615987761658

    if best_map > THRESHOLD:
        inference(model, test_loader, DEVICE)
    else:
        print(
            f"Validation mAP ({best_map}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
