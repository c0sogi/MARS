import os
import time
import torch
import numpy as np
import torch.optim as optim
from library.config import Config
from library.utils import set_seed, save_checkpoint
from library.losses import DeepSupervisionLoss
from library.model import GMG_CRGN
from library.data_loader import get_dataloaders


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0.0
    metrics_sum = {}
    num_batches = 0

    for batch_idx, (features, targets, lengths, mask, _) in enumerate(loader):
        features = features.to(device)
        targets = targets.to(device)
        lengths = lengths.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()

        # Forward pass
        # outputs is a list of tensors [Stage1_Logits, Stage2_Logits, ...]
        outputs = model(features, mask)

        # Compute loss
        loss, batch_metrics = criterion(outputs, targets, lengths)

        # Backward pass
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)

        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        for k, v in batch_metrics.items():
            metrics_sum[k] = metrics_sum.get(k, 0.0) + v

        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in metrics_sum.items()}

    return avg_loss, avg_metrics


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    total_loss = 0.0
    metrics_sum = {}
    correct_frames = 0
    total_valid_frames = 0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, (features, targets, lengths, mask, _) in enumerate(loader):
            features = features.to(device)
            targets = targets.to(device)
            lengths = lengths.to(device)
            mask = mask.to(device)

            # Forward pass
            outputs = model(features, mask)

            # Compute loss
            loss, batch_metrics = criterion(outputs, targets, lengths)

            # Accumulate loss metrics
            total_loss += loss.item()
            for k, v in batch_metrics.items():
                metrics_sum[k] = metrics_sum.get(k, 0.0) + v

            # Calculate Frame Accuracy using the final stage output
            # Final stage output shape: (B, T, C+1)
            final_stage_logits = outputs[-1]
            cls_logits = final_stage_logits[:, :, : Config.NUM_CLASSES]  # (B, T, 21)

            # Get predictions
            preds = torch.argmax(cls_logits, dim=2)  # (B, T)

            # Mask out padding for accuracy calculation
            # mask is True for padding
            valid_mask = ~mask

            batch_correct = (preds == targets) & valid_mask
            correct_frames += batch_correct.sum().item()
            total_valid_frames += valid_mask.sum().item()

            num_batches += 1

    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in metrics_sum.items()}
    accuracy = correct_frames / (total_valid_frames + 1e-8)

    return avg_loss, avg_metrics, accuracy


def run_training(debug=False, load_cached_data=True):
    """
    Main function to run the training pipeline.

    Args:
        debug (bool): If True, uses a small subset of data.
        load_cached_data (bool): If True, attempts to load pre-processed data from cache.
    """
    # Set reproducibility
    set_seed(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Initialize Model
    model = GMG_CRGN().to(device)

    # Initialize Loss
    criterion = DeepSupervisionLoss().to(device)

    # Initialize Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    # Get DataLoaders
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        debug=debug,
        load_cached_data=load_cached_data,
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_metrics, val_acc = validate(model, val_loader, criterion, device)

        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        print(
            f"Epoch: {epoch+1}/{Config.EPOCHS} | Time: {int(epoch_mins)}m {int(epoch_secs)}s"
        )
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Accuracy: {val_acc}")
        # Print detailed metrics for debugging/monitoring
        # print(f"Train Metrics: {train_metrics}")
        # print(f"Val Metrics: {val_metrics}")

        # Checkpoint & Early Stopping
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            print("New best model found!")
        else:
            patience_counter += 1
            print(f"EarlyStopping counter: {patience_counter} out of {Config.PATIENCE}")

        # Save checkpoint
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_score": best_val_loss,
                "val_acc": val_acc,
            },
            is_best=is_best,
            filename=f"checkpoint_epoch_{epoch+1}.pth",
        )

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print("Training complete.")
    return best_val_loss
