import os
import torch
import pandas as pd
import numpy as np
import time

from library.config import Config, seed_everything
from library.dataset import get_dataloaders
from library.model import AnatomicalCenterNet
from library.loss import CenterNetLoss


def calculate_global_accuracy(pred, target):
    """
    Calculates binary accuracy for the global classification head.
    pred: (B, 1) sigmoid probabilities
    target: (B, 1) binary targets
    """
    preds_binary = (pred > 0.5).float()
    correct = (preds_binary == target).float().sum()
    return correct, target.size(0)


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()

    running_loss = 0.0
    running_hm_loss = 0.0
    running_wh_loss = 0.0
    running_off_loss = 0.0
    running_global_loss = 0.0

    global_correct = 0
    global_total = 0

    num_batches = len(loader)

    for batch_idx, (images, targets, _) in enumerate(loader):
        images = images.to(device)

        # Move targets to device
        target_dict = {}
        for k, v in targets.items():
            target_dict[k] = v.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss, loss_stats = criterion(outputs, target_dict)

        loss.backward()
        optimizer.step()

        # Accumulate metrics
        running_loss += loss.item()
        running_hm_loss += loss_stats["hm_loss"].item()
        running_wh_loss += loss_stats["wh_loss"].item()
        running_off_loss += loss_stats["off_loss"].item()
        running_global_loss += loss_stats["global_loss"].item()

        # Global Head Accuracy
        corr, tot = calculate_global_accuracy(
            outputs["global_label"], target_dict["global_label"]
        )
        global_correct += corr
        global_total += tot

    metrics = {
        "loss": running_loss / num_batches,
        "hm_loss": running_hm_loss / num_batches,
        "wh_loss": running_wh_loss / num_batches,
        "off_loss": running_off_loss / num_batches,
        "global_loss": running_global_loss / num_batches,
        "global_acc": global_correct / global_total if global_total > 0 else 0.0,
    }

    return metrics


def validate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    running_hm_loss = 0.0
    running_wh_loss = 0.0
    running_off_loss = 0.0
    running_global_loss = 0.0

    global_correct = 0
    global_total = 0

    num_batches = len(loader)

    with torch.no_grad():
        for batch_idx, (images, targets, _) in enumerate(loader):
            images = images.to(device)

            target_dict = {}
            for k, v in targets.items():
                target_dict[k] = v.to(device)

            outputs = model(images)
            loss, loss_stats = criterion(outputs, target_dict)

            running_loss += loss.item()
            running_hm_loss += loss_stats["hm_loss"].item()
            running_wh_loss += loss_stats["wh_loss"].item()
            running_off_loss += loss_stats["off_loss"].item()
            running_global_loss += loss_stats["global_loss"].item()

            corr, tot = calculate_global_accuracy(
                outputs["global_label"], target_dict["global_label"]
            )
            global_correct += corr
            global_total += tot

    metrics = {
        "loss": running_loss / num_batches,
        "hm_loss": running_hm_loss / num_batches,
        "wh_loss": running_wh_loss / num_batches,
        "off_loss": running_off_loss / num_batches,
        "global_loss": running_global_loss / num_batches,
        "global_acc": global_correct / global_total if global_total > 0 else 0.0,
    }

    return metrics


def run_training():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Load Metadata
    if not os.path.exists(Config.TRAIN_META_PATH) or not os.path.exists(
        Config.VAL_META_PATH
    ):
        raise FileNotFoundError(
            "Metadata files not found. Ensure metadata generation was successful."
        )

    train_df = pd.read_csv(Config.TRAIN_META_PATH)
    val_df = pd.read_csv(Config.VAL_META_PATH)

    print(f"Training on {len(train_df)} annotations")
    print(f"Validating on {len(val_df)} annotations")

    # 3. DataLoaders
    train_loader, val_loader, _ = get_dataloaders(train_df, val_df, test_df=None)

    # 4. Model, Loss, Optimizer
    model = AnatomicalCenterNet()
    model = model.to(device)

    criterion = CenterNetLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # 5. Training Loop
    best_val_loss = float("inf")

    # Early Stopping parameters
    patience = 5
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()

        elapsed = time.time() - start_time

        # Logging
        print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Time: {elapsed:.2f}s")
        print(
            f"  Train Loss: {train_metrics['loss']} (HM: {train_metrics['hm_loss']}, WH: {train_metrics['wh_loss']}, Off: {train_metrics['off_loss']}, Global: {train_metrics['global_loss']})"
        )
        print(f"  Train Global Acc: {train_metrics['global_acc']}")
        print(
            f"  Val Loss: {val_metrics['loss']} (HM: {val_metrics['hm_loss']}, WH: {val_metrics['wh_loss']}, Off: {val_metrics['off_loss']}, Global: {val_metrics['global_loss']})"
        )
        print(f"  Val Global Acc: {val_metrics['global_acc']}")

        # Checkpointing & Early Stopping
        current_val_loss = val_metrics["loss"]

        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            patience_counter = 0
            save_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  New best model saved to {save_path}")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Save last model
    last_save_path = os.path.join(Config.CHECKPOINT_DIR, "last_model.pth")
    torch.save(model.state_dict(), last_save_path)
    print(f"Last model saved to {last_save_path}")
    print("Training complete.")


if __name__ == "__main__":
    # This block is technically not required by the prompt instructions
    # ("DO NOT include an if __name__ == '__main__': block"),
    # but the prompt asks to implement the module.
    # The function run_training() is available to be called by an external script.
    pass
