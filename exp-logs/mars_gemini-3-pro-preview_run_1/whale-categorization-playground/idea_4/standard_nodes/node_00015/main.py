import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.stats import pearsonr

# Import from the provided library files
from library.config import Config
from library.utils import (
    seed_everything,
    save_checkpoint,
    load_checkpoint,
    map_per_image,
)
from library.dataset import get_dataloaders
from library.model import WhaleConvNeXt
from library.train import train_one_epoch
from library.evaluate import validate, inference


def run():
    # -------------------------------------------------------------------------
    # 1. Setup & Initialization
    # -------------------------------------------------------------------------
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("Loading Data...")
    # Load data with caching enabled for speed
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        load_cached_data=True
    )

    print(f"  Train Set: {len(train_loader.dataset)} images")
    print(f"  Val Set:   {len(val_loader.dataset)} images")
    print(f"  Test Set:  {len(test_loader.dataset)} images")
    print(f"  Classes:   {len(classes)}")

    # -------------------------------------------------------------------------
    # 3. Model & Optimizer
    # -------------------------------------------------------------------------
    print(f"Initializing Model: {Config.BACKBONE} + ArcFace")
    model = WhaleConvNeXt()
    model = model.to(device)

    # Loss: CrossEntropyLoss works with the angular margin logits from ArcFace
    criterion = nn.CrossEntropyLoss()

    # Optimizer: AdamW
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: Cosine Annealing
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR)

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    print(f"Starting Training for {Config.EPOCHS} epochs...")
    best_map5 = 0.0

    for epoch in range(1, Config.EPOCHS + 1):
        epoch_start = time.time()

        # --- Train ---
        train_loss = train_one_epoch(
            train_loader, model, criterion, optimizer, device, epoch
        )

        # --- Validate (with TTA) ---
        val_loss, val_map5 = validate(val_loader, model, criterion, device, classes)

        # --- Scheduler Step ---
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # --- Logging ---
        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch}/{Config.EPOCHS} | Time: {elapsed:.1f}s | LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAP@5: {val_map5:.6f}"
        )

        # --- Checkpointing ---
        is_best = val_map5 > best_map5
        if is_best:
            best_map5 = val_map5
            print(f"  >>> Found new best model! MAP@5: {best_map5:.6f}")

        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_map5": best_map5,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
        )

    print(f"Training Complete. Best MAP@5: {best_map5:.6f}")

    # -------------------------------------------------------------------------
    # 5. Failure Analysis
    # -------------------------------------------------------------------------
    print("\nRunning Failure Analysis on Best Model...")

    # Load the best checkpoint
    checkpoint = load_checkpoint(model, filename="model_best.pth.tar", device=device)
    if checkpoint is None:
        print("Warning: Best checkpoint not found. Using current model.")

    model.eval()

    all_aps = []
    feature_means = []
    feature_stds = []

    # Iterate through validation set to compute per-sample metrics
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            # TTA: Horizontal Flip
            logits_orig = model(images, labels=None)
            images_flipped = torch.flip(images, dims=[3])
            logits_flip = model(images_flipped, labels=None)

            # Average logits
            logits = (logits_orig + logits_flip) / 2.0

            # Get predictions
            _, top_indices = torch.topk(logits, k=5, dim=1)
            top_indices = top_indices.cpu().numpy()
            labels_np = labels.cpu().numpy()

            # Process batch
            for i in range(len(labels_np)):
                # 1. Calculate AP for this image
                pred_labels = [classes[idx] for idx in top_indices[i]]
                true_label = classes[labels_np[i]]
                ap = map_per_image(pred_labels, true_label)
                all_aps.append(ap)

                # 2. Extract Input Features (from normalized tensor)
                # Image shape: (3, H, W)
                img_tensor = images[i]
                feature_means.append(torch.mean(img_tensor).item())
                feature_stds.append(torch.std(img_tensor).item())

    # Calculate Final Metric
    final_metric = np.mean(all_aps)
    print(f"Final Validation Metric: {final_metric}")

    # Calculate Correlations
    # Error Magnitude = 1.0 - AP (0 is perfect, 1 is total failure)
    errors = 1.0 - np.array(all_aps)
    means = np.array(feature_means)
    stds = np.array(feature_stds)

    # Pearson Correlation
    # Check for non-zero variance to avoid warnings
    if np.std(errors) > 1e-9 and np.std(means) > 1e-9:
        corr_mean, _ = pearsonr(errors, means)
    else:
        corr_mean = 0.0

    if np.std(errors) > 1e-9 and np.std(stds) > 1e-9:
        corr_std, _ = pearsonr(errors, stds)
    else:
        corr_std = 0.0

    print("Correlation between Error Magnitude (1-AP) and Input Features:")
    print(f"  Mean Intensity: {corr_mean:.4f}")
    print(f"  Pixel Std Dev:  {corr_std:.4f}")

    # -------------------------------------------------------------------------
    # 6. Submission
    # -------------------------------------------------------------------------
    THRESHOLD = 0.6545824094604581

    if final_metric > THRESHOLD:
        print(
            f"\nPerformance ({final_metric:.6f}) exceeds threshold ({THRESHOLD:.6f})."
        )
        inference(test_loader, model, device, classes)
    else:
        print(
            f"\nPerformance ({final_metric:.6f}) does not meet threshold ({THRESHOLD:.6f}). Submission skipped."
        )


if __name__ == "__main__":
    run()
