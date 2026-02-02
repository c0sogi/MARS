import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_loaders
from library.model import ResNet18D_UNet
from library.engine import train_one_epoch, validate
from library.inference import predict_and_submit


def analyze_failures(model, data_loader, device):
    """
    Performs failure analysis on the validation set.
    Calculates the correlation between error magnitude and input features.
    """
    print("\n==== Failure Analysis ====")
    model.eval()

    errors = []
    class_indices = []
    mask_areas = []

    # Define loss functions with reduction='none' to get per-sample loss
    criterion_cls = nn.CrossEntropyLoss(reduction="none")
    criterion_seg = nn.BCEWithLogitsLoss(reduction="none")

    with torch.no_grad():
        for images, labels, masks in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            # Forward pass
            logits, pred_masks = model(images)

            # 1. Calculate Error Magnitude (Loss)
            targets_idx = torch.argmax(labels, dim=1)
            loss_cls = criterion_cls(logits, targets_idx)  # Shape: (B,)

            # Segmentation loss: Average over pixels for each image
            # masks shape: (B, 1, H, W)
            # loss_seg_map shape: (B, 1, H, W)
            loss_seg_map = criterion_seg(pred_masks, masks)
            loss_seg = loss_seg_map.mean(dim=(1, 2, 3))  # Shape: (B,)

            # Weighted composite error
            total_error = (Config.lambda_cls * loss_cls) + (
                Config.lambda_seg * loss_seg
            )

            errors.extend(total_error.cpu().tolist())
            class_indices.extend(targets_idx.cpu().tolist())

            # 2. Input Features
            # Feature: Mask Area (proxy for opacity size/count)
            # Sum of binary mask pixels
            current_mask_areas = masks.sum(dim=(1, 2, 3)).cpu().tolist()
            mask_areas.extend(current_mask_areas)

    # Create DataFrame for analysis
    df = pd.DataFrame(
        {
            "error_magnitude": errors,
            "class_index": class_indices,
            "mask_area": mask_areas,
        }
    )

    # Calculate correlations
    # We correlate error_magnitude with other features
    correlation_matrix = df.corr()
    error_correlations = correlation_matrix["error_magnitude"].drop("error_magnitude")

    print("Correlation between Error Magnitude and Input Features:")
    print(error_correlations)

    return df


def main():
    # 1. Setup
    seed_everything(Config.seed)
    device = Config.device

    # Ensure we use the full dataset for the best score
    Config.debug = False

    print(f"Starting execution on device: {device}")

    # 2. Data Loading
    print("Loading datasets...")
    train_loader, val_loader, _ = get_loaders(debug=Config.debug, load_cached_data=True)

    # 3. Model Initialization
    print("Initializing model...")
    model = ResNet18D_UNet(num_classes=Config.num_study_classes, pretrained=True)
    model.to(device)

    # Optimization
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.epochs, eta_min=Config.min_lr
    )

    # 4. Training Loop
    best_map = 0.0
    best_model_path = os.path.join(Config.cache_dir, "best_model.pth")

    print(f"Starting training for {Config.epochs} epochs...")

    for epoch in range(1, Config.epochs + 1):
        print(f"\nEpoch {epoch}/{Config.epochs}")

        # Train
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)

        # Validate
        val_loss, val_map = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        # Checkpointing
        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), best_model_path)
            print(f"New Best mAP: {best_map:.6f} (Saved to {best_model_path})")

    # 5. Final Reporting
    print(f"\nTraining Complete.")
    print(f"Final Validation Metric: {best_map}")

    # 6. Failure Analysis
    if os.path.exists(best_model_path):
        print("Loading best model for analysis...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        analyze_failures(model, val_loader, device)

    # 7. Submission
    THRESHOLD = 0.49944536565378

    if best_map > THRESHOLD:
        print(f"\nValidation mAP ({best_map}) exceeds threshold ({THRESHOLD}).")
        print("Generating submission...")
        predict_and_submit(model_path=best_model_path)
    else:
        print(f"\nValidation mAP ({best_map}) does not exceed threshold ({THRESHOLD}).")
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
