import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from library.dataset import SaltDataset
from library.model import DeepResUNet
from library.loss import CompoundLoss
from library.utils import seed_everything, calculate_iou_map, rle_encode


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Training loop for one epoch.
    Handles Deep Supervision outputs and weighted compound loss.
    """
    model.train()
    running_loss = 0.0
    dataset_size = len(loader.dataset)

    for images, masks, depths, _ in loader:
        images = images.to(device)
        masks = masks.to(device)
        depths = depths.to(device)

        optimizer.zero_grad()

        # Forward pass: returns (final, aux1, aux2) in training mode
        final_out, aux1, aux2 = model(images, depths)

        # Compute loss for each head
        # The model outputs are already upsampled/cropped to match the input size (128x128)
        loss_final = criterion(final_out, masks)
        loss_aux1 = criterion(aux1, masks)
        loss_aux2 = criterion(aux2, masks)

        # Weighted sum for Deep Supervision
        loss = loss_final + 0.5 * loss_aux1 + 0.25 * loss_aux2

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / dataset_size


def validate(model, loader, device):
    """
    Validation loop.
    Calculates mAP on original 101x101 size.
    """
    model.eval()
    all_preds = []
    all_masks = []

    # Crop indices to restore 101x101 from 128x128 padded input
    # Padding was 27 pixels total: 13 left/top, 14 right/bottom
    start_idx = 13
    end_idx = 114

    with torch.no_grad():
        for images, masks, depths, _ in loader:
            images = images.to(device)
            masks = masks.to(device)
            depths = depths.to(device)

            # Forward pass: returns final_out only in eval mode
            logits = model(images, depths)
            preds = torch.sigmoid(logits)

            # Crop to original size for accurate metric calculation
            preds_cropped = preds[..., start_idx:end_idx, start_idx:end_idx]
            masks_cropped = masks[..., start_idx:end_idx, start_idx:end_idx]

            all_preds.append(preds_cropped.cpu())
            all_masks.append(masks_cropped.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    # Calculate Mean Average Precision
    score = calculate_iou_map(all_preds, all_masks)
    return score


def generate_submission(model_paths, work_dir, device):
    """
    Generates submission file using Snapshot Ensembling and TTA.
    """
    print("Generating submission with Snapshot Ensemble...")

    # Load Test Data
    test_dataset = SaltDataset(mode="test", work_dir=work_dir)
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

    # Load Models
    models = []
    for path in model_paths:
        if os.path.exists(path):
            print(f"Loading checkpoint: {path}")
            m = DeepResUNet(in_channels=1, out_channels=1).to(device)
            m.load_state_dict(torch.load(path, map_location=device))
            m.eval()
            models.append(m)
        else:
            print(f"Warning: Checkpoint {path} not found. Skipping.")

    if not models:
        print("No models loaded. Cannot generate submission.")
        return

    submission_data = []

    # Crop indices for 101x101
    start_idx = 13
    end_idx = 114

    with torch.no_grad():
        for images, _, depths, ids in test_loader:
            images = images.to(device)
            depths = depths.to(device)

            batch_preds = 0.0

            # Ensemble + TTA (Horizontal Flip)
            for model in models:
                # Original
                out = torch.sigmoid(model(images, depths))
                batch_preds += out

                # Flip TTA
                images_flipped = torch.flip(images, dims=[3])
                out_flipped = torch.sigmoid(model(images_flipped, depths))
                batch_preds += torch.flip(out_flipped, dims=[3])

            # Average: Sum / (Num_Models * 2_augmentations)
            batch_preds /= len(models) * 2

            # Crop to 101x101
            batch_preds = batch_preds[..., start_idx:end_idx, start_idx:end_idx]

            # Binarize (Threshold 0.5)
            batch_masks = (batch_preds > 0.5).byte().cpu().numpy()

            for i, img_id in enumerate(ids):
                mask = batch_masks[i, 0]
                rle = rle_encode(mask)
                submission_data.append({"id": img_id, "rle_mask": rle})

    # Save CSV
    df_sub = pd.DataFrame(submission_data)
    os.makedirs("./submission", exist_ok=True)
    sub_path = "./submission/submission.csv"
    df_sub.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")


def run_training(
    batch_size=32,
    epochs=150,
    lr=1e-3,
    work_dir="./working/idea_9",
    save_dir="./working/idea_9/checkpoints",
):
    """
    Main training function implementing the cyclic schedule and snapshot saving.
    """
    seed_everything(42)
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    print("Loading datasets...")
    train_dataset = SaltDataset(mode="train", work_dir=work_dir)
    val_dataset = SaltDataset(mode="val", work_dir=work_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # --- Model Setup ---
    print("Initializing model...")
    model = DeepResUNet(in_channels=1, out_channels=1).to(device)

    criterion = CompoundLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Cosine Annealing Warm Restarts: T_0=50 epochs, T_mult=1
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=1e-6)

    # --- Training Loop ---
    best_map_cycle_2 = -1.0
    best_map_cycle_3 = -1.0

    print("Starting training...")
    start_time = time.time()

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_map = validate(model, val_loader, device)

        # Step scheduler at epoch end
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1}/{epochs} | LR: {current_lr:.6f} | Train Loss: {train_loss:.5f} | Val mAP: {val_map}"
        )

        # Snapshot Logic:
        # Cycle 2: Epochs 51-100 (indices 50-99)
        if 50 <= epoch < 100:
            if val_map > best_map_cycle_2:
                best_map_cycle_2 = val_map
                torch.save(
                    model.state_dict(), os.path.join(save_dir, "best_cycle_2.pth")
                )

        # Cycle 3: Epochs 101-150 (indices 100-149)
        if 100 <= epoch < 150:
            if val_map > best_map_cycle_3:
                best_map_cycle_3 = val_map
                torch.save(
                    model.state_dict(), os.path.join(save_dir, "best_cycle_3.pth")
                )

    total_time = time.time() - start_time
    print(f"Training finished in {total_time/3600:.2f} hours.")
    print(f"Best Cycle 2 mAP: {best_map_cycle_2}")
    print(f"Best Cycle 3 mAP: {best_map_cycle_3}")

    # Generate Submission using the snapshots
    generate_submission(
        model_paths=[
            os.path.join(save_dir, "best_cycle_2.pth"),
            os.path.join(save_dir, "best_cycle_3.pth"),
        ],
        work_dir=work_dir,
        device=device,
    )
