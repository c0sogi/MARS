import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn as nn

# Import from library
from library.utils import set_seed, rle_encode, compute_map_score
from library.losses import BCEDiceLoss, DeepSupervisionLoss
from library.model import DeepResUNet
from library.dataset import get_dataloaders
from library.engine import train_one_epoch, validate, center_crop

# Configuration
SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 150
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "./working/checkpoints"
SUBMISSION_DIR = "./submission"
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure directories exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)


def main():
    set_seed(SEED)

    # 1. Data Loading
    print("Loading data...")
    # Disable cache loading to ensure full dataset is processed
    train_loader, val_loader, test_loader = get_dataloaders(
        train_csv_path="./metadata/train.csv",
        val_csv_path="./metadata/val.csv",
        test_csv_path="./metadata/test.csv",
        batch_size=BATCH_SIZE,
        num_workers=2,
        load_cached_data=False,
    )

    # 2. Model Initialization
    print(f"Initializing DeepResUNet on {DEVICE}...")
    model = DeepResUNet(in_channels=1, out_channels=1, depth_fused=True).to(DEVICE)

    # 3. Optimization
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # Restart every 50 epochs (Cycles at 50, 100, 150)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=1e-5)

    # 4. Loss Functions
    # Use corrected BCEDiceLoss with Deep Supervision throughout all cycles
    criterion = DeepSupervisionLoss(
        BCEDiceLoss(bce_weight=0.5, dice_weight=0.5),
        weights=[1.0, 0.5, 0.25],  # Weights for 128x128, 64x64, 32x32 heads
    )

    # Training State
    best_map_cycle1 = 0.0
    best_map_cycle2 = 0.0
    best_map_cycle3 = 0.0

    path_cycle1 = os.path.join(CHECKPOINT_DIR, "best_cycle_1.pth")
    path_cycle2 = os.path.join(CHECKPOINT_DIR, "best_cycle_2.pth")
    path_cycle3 = os.path.join(CHECKPOINT_DIR, "best_cycle_3.pth")

    print(f"Starting training for {NUM_EPOCHS} epochs...")

    for epoch in range(1, NUM_EPOCHS + 1):
        # Determine Cycle
        if epoch <= 50:
            cycle_idx = 1
        elif epoch <= 100:
            cycle_idx = 2
        else:
            cycle_idx = 3

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE, epoch
        )

        # Validate
        val_loss, val_map = validate(model, val_loader, criterion, DEVICE)

        # Scheduler Step
        scheduler.step()

        print(
            f"Epoch {epoch} | Cycle {cycle_idx} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mAP: {val_map:.4f}"
        )

        # Snapshot Logic - Save Best of Each Cycle
        if cycle_idx == 1:
            if val_map > best_map_cycle1:
                best_map_cycle1 = val_map
                torch.save(model.state_dict(), path_cycle1)
                print(f"--> Saved Cycle 1 Best: {val_map:.4f}")
        elif cycle_idx == 2:
            if val_map > best_map_cycle2:
                best_map_cycle2 = val_map
                torch.save(model.state_dict(), path_cycle2)
                print(f"--> Saved Cycle 2 Best: {val_map:.4f}")
        elif cycle_idx == 3:
            if val_map > best_map_cycle3:
                best_map_cycle3 = val_map
                torch.save(model.state_dict(), path_cycle3)
                print(f"--> Saved Cycle 3 Best: {val_map:.4f}")

    # 5. Ensemble Inference & Failure Analysis
    print("\nStarting Ensemble Validation & Failure Analysis...")

    # Load snapshots
    models = []
    checkpoints = [path_cycle1, path_cycle2, path_cycle3]

    for ckpt_path in checkpoints:
        if os.path.exists(ckpt_path):
            m = DeepResUNet(in_channels=1, out_channels=1, depth_fused=True).to(DEVICE)
            m.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            m.eval()
            models.append(m)
        else:
            print(f"Warning: Checkpoint {ckpt_path} not found. Skipping.")

    if not models:
        print("No checkpoints found. Using current model state.")
        model.eval()
        models.append(model)

    # Validation Inference
    val_preds = []
    val_targets = []
    val_ids = []

    with torch.no_grad():
        for images, masks, ids in val_loader:
            images = images.to(DEVICE)

            # Ensemble predictions
            batch_preds = []
            for m in models:
                # TTA Forward
                l = m(images)
                l_flip = torch.flip(m(torch.flip(images, [3])), [3])
                p = (torch.sigmoid(l) + torch.sigmoid(l_flip)) / 2.0
                batch_preds.append(p)

            # Average across models
            p_ens = torch.stack(batch_preds).mean(dim=0)

            # Crop
            p_ens = center_crop(p_ens)
            masks = center_crop(masks)

            val_preds.append(p_ens.cpu().numpy())
            val_targets.append(masks.cpu().numpy())
            val_ids.extend(ids)

    val_preds_np = np.concatenate(val_preds, axis=0)
    val_targets_np = np.concatenate(val_targets, axis=0)

    # Compute Final Metric
    final_metric = compute_map_score(val_preds_np, val_targets_np)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    # Calculate per-image IoU to approximate error
    preds_bin = (val_preds_np > 0.5).astype(np.uint8).squeeze()
    targets_bin = (val_targets_np > 0.5).astype(np.uint8).squeeze()

    inter = (preds_bin & targets_bin).sum(axis=(1, 2))
    union = (preds_bin | targets_bin).sum(axis=(1, 2))
    # Avoid div by zero
    ious = np.ones_like(inter, dtype=float)
    non_empty = union > 0
    ious[non_empty] = inter[non_empty] / union[non_empty]
    errors = 1.0 - ious

    # Load metadata to correlate
    df_val = pd.read_csv("./metadata/val.csv")
    # Map errors to ids
    error_map = {id_: err for id_, err in zip(val_ids, errors)}
    df_val["error"] = df_val["id"].map(error_map)

    # Correlations
    print("Failure Analysis Correlations:")
    if "z" in df_val.columns and "error" in df_val.columns:
        corr_depth = df_val["z"].corr(df_val["error"])
        print(f"Correlation (Depth vs Error): {corr_depth}")

    if "coverage" in df_val.columns:
        corr_cov = df_val["coverage"].corr(df_val["error"])
        print(f"Correlation (Salt Coverage vs Error): {corr_cov}")

    # 6. Submission
    threshold_score = 0.8156666666666668
    if final_metric > threshold_score:
        print("Metric threshold passed. Generating submission...")

        test_preds_rle = []
        test_ids_list = []

        if test_loader is not None:
            with torch.no_grad():
                for images, ids in test_loader:
                    images = images.to(DEVICE)

                    # Ensemble predictions
                    batch_preds = []
                    for m in models:
                        l = m(images)
                        l_flip = torch.flip(m(torch.flip(images, [3])), [3])
                        p = (torch.sigmoid(l) + torch.sigmoid(l_flip)) / 2.0
                        batch_preds.append(p)

                    p_ens = torch.stack(batch_preds).mean(dim=0)

                    # Crop
                    p_ens = center_crop(p_ens)  # (B, 1, 101, 101)

                    # Binarize
                    p_bin = (p_ens > 0.5).float().cpu().numpy()

                    for i in range(len(ids)):
                        mask = p_bin[i, 0, :, :]
                        rle = rle_encode(mask)
                        test_preds_rle.append(rle)
                        test_ids_list.append(ids[i])

            # Create DataFrame
            sub_df = pd.DataFrame({"id": test_ids_list, "rle_mask": test_preds_rle})
            sub_df.to_csv(SUBMISSION_FILE, index=False)
            print(f"Submission saved to {SUBMISSION_FILE}")
        else:
            print("Test loader not available (no test metadata).")
    else:
        print(
            f"Metric {final_metric} did not beat threshold {threshold_score}. Submission skipped."
        )


if __name__ == "__main__":
    main()
