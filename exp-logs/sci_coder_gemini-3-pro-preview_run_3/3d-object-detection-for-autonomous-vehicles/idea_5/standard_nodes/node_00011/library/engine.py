import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from library.config import (
    DEVICE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    GRAD_NORM_CLIP,
    NUM_WORKERS,
    SUBMISSION_PATH,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    CLASS_NAMES,
    WORKING_DIR,
    set_deterministic,
    DEBUG_N_SAMPLES,
)
from library.dataset import LidarDataset, collate_fn
from library.model import CenterPointNet
from library.loss import FastFocalLoss, RegLoss
from library.utils import decode_predictions

# Ensure deterministic behavior
set_deterministic()


class LossWrapper(nn.Module):
    """
    Wrapper to calculate and combine losses from different heads.
    """

    def __init__(self):
        super().__init__()
        self.heatmap_loss = FastFocalLoss()
        self.reg_loss = RegLoss()

        # Loss weights
        self.weights = {
            "heatmap": 1.0,
            "dim": 0.1,  # Dimensions are often harder/noisier
            "rot": 1.0,
            "reg": 1.0,  # Local offset
            "z_map": 1.0,
        }

    def forward(self, preds, targets):
        # Heatmap Loss
        hm_loss = self.heatmap_loss(preds["heatmap"], targets["heatmap"])

        # Regression Losses (Masked by object presence)
        # targets["mask"] is (B, K)
        # targets["ind"] is (B, K)
        mask = targets["mask"]
        ind = targets["ind"]

        dim_loss = self.reg_loss(preds["dim"], targets["dim"], mask, ind)
        rot_loss = self.reg_loss(preds["rot"], targets["rot"], mask, ind)
        reg_loss = self.reg_loss(preds["reg"], targets["reg"], mask, ind)
        z_loss = self.reg_loss(preds["z_map"], targets["z_map"], mask, ind)

        # Weighted Sum
        total_loss = (
            self.weights["heatmap"] * hm_loss
            + self.weights["dim"] * dim_loss
            + self.weights["rot"] * rot_loss
            + self.weights["reg"] * reg_loss
            + self.weights["z_map"] * z_loss
        )

        loss_stats = {
            "total_loss": total_loss.item(),
            "hm_loss": hm_loss.item(),
            "dim_loss": dim_loss.item(),
            "rot_loss": rot_loss.item(),
            "reg_loss": reg_loss.item(),
            "z_loss": z_loss.item(),
        }

        return total_loss, loss_stats


def train_one_epoch(
    model, dataloader, optimizer, scheduler, criterion, device, epoch_idx
):
    model.train()
    running_stats = {}
    num_batches = len(dataloader)

    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        # Move data to device
        points = [p.to(device) for p in batch["points"]]

        # Stacked tensors
        targets = {
            "heatmap": batch["heatmap"].to(device),
            "dim": batch["dim"].to(device),
            "rot": batch["rot"].to(device),
            "reg": batch["reg"].to(device),
            "z_map": batch["z_map"].to(device),
            "ind": batch["ind"].to(device),
            "mask": batch["mask"].to(device),
        }

        optimizer.zero_grad()

        # Forward
        preds = model({"points": points})

        # Loss
        loss, stats = criterion(preds, targets)

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_NORM_CLIP)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Accumulate stats
        for k, v in stats.items():
            running_stats[k] = running_stats.get(k, 0.0) + v

    # Average stats
    avg_stats = {k: v / num_batches for k, v in running_stats.items()}
    epoch_time = time.time() - start_time

    print(
        f"Epoch {epoch_idx+1} Train | Time: {epoch_time:.1f}s | "
        f"Loss: {avg_stats['total_loss']:.6f} | "
        f"HM: {avg_stats['hm_loss']:.4f} | "
        f"Dim: {avg_stats['dim_loss']:.4f} | "
        f"Reg: {avg_stats['reg_loss']:.4f}"
    )

    return avg_stats


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_stats = {}
    num_batches = len(dataloader)

    for batch in dataloader:
        points = [p.to(device) for p in batch["points"]]

        targets = {
            "heatmap": batch["heatmap"].to(device),
            "dim": batch["dim"].to(device),
            "rot": batch["rot"].to(device),
            "reg": batch["reg"].to(device),
            "z_map": batch["z_map"].to(device),
            "ind": batch["ind"].to(device),
            "mask": batch["mask"].to(device),
        }

        preds = model({"points": points})
        loss, stats = criterion(preds, targets)

        for k, v in stats.items():
            running_stats[k] = running_stats.get(k, 0.0) + v

    avg_stats = {k: v / num_batches for k, v in running_stats.items()}

    print(
        f"Validation | Loss: {avg_stats['total_loss']:.6f} | "
        f"HM: {avg_stats['hm_loss']:.4f}"
    )

    return avg_stats


def train_model():
    print("Initializing Datasets...")
    train_dataset = LidarDataset(
        TRAIN_METADATA_PATH, mode="train", num_samples=DEBUG_N_SAMPLES
    )
    val_dataset = LidarDataset(
        VAL_METADATA_PATH, mode="val", num_samples=DEBUG_N_SAMPLES
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print(f"Train Samples: {len(train_dataset)} | Val Samples: {len(val_dataset)}")

    # Model Setup
    model = CenterPointNet().to(DEVICE)
    criterion = LossWrapper().to(DEVICE)

    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # One Cycle Scheduler
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        total_steps=total_steps,
        pct_start=0.3,
        div_factor=10,
        final_div_factor=100,
    )

    # Training Loop
    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0
    checkpoint_path = os.path.join(WORKING_DIR, "best_model.pth")

    print("\nStarting Training...")
    for epoch in range(NUM_EPOCHS):
        train_stats = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, DEVICE, epoch
        )
        val_stats = evaluate(model, val_loader, criterion, DEVICE)

        val_loss = val_stats["total_loss"]

        # Checkpoint & Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"--> Best model saved (Val Loss: {best_val_loss:.6f})")
        else:
            patience_counter += 1
            print(f"--> No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    return checkpoint_path


def generate_submission(model_path):
    print("\nGenerating Submission...")

    # Load Test Data
    test_dataset = LidarDataset(TEST_METADATA_PATH, mode="test", num_samples=None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # Load Model
    model = CenterPointNet().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    results = []
    confidence_threshold = 0.1  # Filter low confidence predictions

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            points = [p.to(device=DEVICE) for p in batch["points"]]
            sample_tokens = batch["sample_tokens"]

            # Forward
            preds = model({"points": points})

            # Decode
            # Returns (B, K, 9) -> [x, y, z, w, l, h, yaw, score, class_id]
            detections = decode_predictions(
                preds["heatmap"],
                preds["dim"],
                preds["rot"],
                preds["reg"],
                preds["z_map"],
                K=50,
            )

            detections = detections.cpu().numpy()

            for i, token in enumerate(sample_tokens):
                sample_dets = detections[i]

                prediction_strings = []
                for det in sample_dets:
                    x, y, z, w, l, h, yaw, score, cls_id = det

                    if score < confidence_threshold:
                        continue

                    cls_name = CLASS_NAMES[int(cls_id)]

                    # Format: score x y z w l h yaw class_name
                    # Note: Submission example uses space delimited
                    pred_str = f"{score:.4f} {x:.4f} {y:.4f} {z:.4f} {w:.4f} {l:.4f} {h:.4f} {yaw:.4f} {cls_name}"
                    prediction_strings.append(pred_str)

                full_pred_str = " ".join(prediction_strings)
                results.append({"Id": token, "PredictionString": full_pred_str})

    # Save CSV
    df = pd.DataFrame(results)

    # Ensure directory exists
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")


if __name__ == "__main__":
    # Train
    best_model_path = train_model()

    # Predict
    generate_submission(best_model_path)
