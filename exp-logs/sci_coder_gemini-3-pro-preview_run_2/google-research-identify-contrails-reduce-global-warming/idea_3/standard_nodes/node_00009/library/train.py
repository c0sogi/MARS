import os
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from library.config import Config
from library.utils import set_seed, rle_encode
from library.dataset import ContrailDataset
from library.model import MultiTaskResNetUNet
from library.loss import MultiTaskLoss


def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    running_seg_loss = 0.0
    running_cls_loss = 0.0

    dataset_size = 0

    for images, masks, labels, _ in loader:
        batch_size = images.size(0)

        images = images.to(device)
        masks = masks.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        seg_logits, cls_logits = model(images)

        # Compute loss
        loss_dict = criterion(seg_logits, cls_logits, masks, labels)
        loss = loss_dict["loss"]

        # Backward pass
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        # Accumulate metrics
        running_loss += loss.item() * batch_size
        running_seg_loss += loss_dict["seg_loss"].item() * batch_size
        running_cls_loss += loss_dict["cls_loss"].item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    epoch_seg_loss = running_seg_loss / dataset_size
    epoch_cls_loss = running_cls_loss / dataset_size

    return epoch_loss, epoch_seg_loss, epoch_cls_loss


def validate(model, loader, criterion, device):
    """
    Validates the model on the validation set.
    Computes Global Dice Coefficient using Gated Inference.
    """
    model.eval()
    running_loss = 0.0

    # Global Dice Accumulators
    intersection_sum = 0.0
    union_sum = 0.0

    # Classification Accuracy Accumulator
    correct_cls = 0
    total_cls = 0

    dataset_size = 0

    with torch.no_grad():
        for images, masks, labels, _ in loader:
            batch_size = images.size(0)

            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            # Forward pass
            seg_logits, cls_logits = model(images)

            # Loss calculation
            loss_dict = criterion(seg_logits, cls_logits, masks, labels)
            running_loss += loss_dict["loss"].item() * batch_size
            dataset_size += batch_size

            # --------------------------------------------------
            # Metrics Calculation
            # --------------------------------------------------

            # 1. Classification Accuracy
            cls_probs = torch.sigmoid(cls_logits)
            cls_preds = (cls_probs > Config.CLS_THRESHOLD).float()
            # Ensure labels are (B, 1)
            if labels.ndim == 1:
                labels = labels.view(-1, 1)
            correct_cls += (cls_preds == labels).sum().item()
            total_cls += batch_size

            # 2. Global Dice with Gated Inference
            seg_probs = torch.sigmoid(seg_logits)

            # Pixel-level thresholding
            pred_masks = (seg_probs > Config.PIXEL_THRESHOLD).float()

            # Apply Classification Gate
            # If image-level prob < threshold, zero out the entire mask for that image
            # cls_probs is (B, 1), pred_masks is (B, 1, H, W)
            gate_mask = (
                (cls_probs > Config.CLS_THRESHOLD).float().view(batch_size, 1, 1, 1)
            )
            pred_masks = pred_masks * gate_mask

            # Flatten for global intersection/union
            pred_flat = pred_masks.view(-1)
            true_flat = masks.view(-1)

            intersection_sum += (pred_flat * true_flat).sum().item()
            union_sum += pred_flat.sum().item() + true_flat.sum().item()

    epoch_loss = running_loss / dataset_size
    cls_accuracy = correct_cls / total_cls if total_cls > 0 else 0.0

    # Global Dice Formula: 2 * |X n Y| / (|X| + |Y|)
    smooth = 1e-6
    global_dice = (2.0 * intersection_sum + smooth) / (union_sum + smooth)

    return epoch_loss, global_dice, cls_accuracy


def train_model(debug=False):
    """
    Main training loop.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Setup directories
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # --------------------------------------------------
    # Data Loading
    # --------------------------------------------------
    # Use smaller sample for debugging if requested
    max_samples = 100 if debug else None

    train_dataset = ContrailDataset(split="train", max_samples=max_samples)
    val_dataset = ContrailDataset(split="validation", max_samples=max_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(
        f"Training on {len(train_dataset)} samples, Validating on {len(val_dataset)} samples."
    )

    # --------------------------------------------------
    # Model Setup
    # --------------------------------------------------
    model = MultiTaskResNetUNet(in_channels=Config.IN_CHANNELS, pretrained=True)
    model.to(device)

    criterion = MultiTaskLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Cosine Annealing Scheduler
    # Note: T_max should ideally be steps per epoch * epochs, but Config.T_MAX is set to EPOCHS.
    # We will step per epoch as per standard CosineAnnealingLR usage if T_max is epochs.
    # However, standard practice with AdamW often steps per batch.
    # Given the Config defines T_MAX = EPOCHS, we assume epoch-based stepping or OneCycle logic.
    # We'll use CosineAnnealingLR stepping once per epoch.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )

    # --------------------------------------------------
    # Training Loop
    # --------------------------------------------------
    best_dice = 0.0
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss, train_seg_loss, train_cls_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scheduler=None
        )

        # Step scheduler at epoch end
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Validate
        val_loss, val_dice, val_acc = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Time: {elapsed:.1f}s | LR: {current_lr:.2e}"
        )
        print(
            f"  Train Loss: {train_loss:.6f} (Seg: {train_seg_loss:.6f}, Cls: {train_cls_loss:.6f})"
        )
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  Val Dice:   {val_dice:.10f}")  # Full precision as requested
        print(f"  Val Cls Acc:{val_acc:.6f}")

        # Early Stopping & Checkpointing
        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  New Best Dice! Model saved to {best_model_path}")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Dice: {best_dice:.10f}")
    return best_model_path


def inference(model_path):
    """
    Generates predictions for the test set using the best model.
    Applies Gated Inference and RLE encoding.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Load Model
    model = MultiTaskResNetUNet(in_channels=Config.IN_CHANNELS, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Test Dataset
    test_dataset = ContrailDataset(split="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    submission_data = []

    print("Starting Inference on Test Set...")

    with torch.no_grad():
        for images, _, _, record_ids in test_loader:
            images = images.to(device)

            # Forward
            seg_logits, cls_logits = model(images)

            # Probabilities
            seg_probs = torch.sigmoid(seg_logits)
            cls_probs = torch.sigmoid(cls_logits)

            # Gated Inference
            # 1. Pixel Threshold
            pred_masks = (seg_probs > Config.PIXEL_THRESHOLD).float()

            # 2. Classification Gate
            gate_mask = (cls_probs > Config.CLS_THRESHOLD).float().view(-1, 1, 1, 1)
            pred_masks = pred_masks * gate_mask

            # Convert to numpy for RLE
            pred_masks_np = pred_masks.squeeze(1).cpu().numpy().astype(np.uint8)

            for i, record_id in enumerate(record_ids):
                mask = pred_masks_np[i]
                rle = rle_encode(mask)
                submission_data.append({"record_id": record_id, "encoded_pixels": rle})

    # Create Submission DataFrame
    df_sub = pd.DataFrame(submission_data)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)

    df_sub.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE} with {len(df_sub)} records.")
