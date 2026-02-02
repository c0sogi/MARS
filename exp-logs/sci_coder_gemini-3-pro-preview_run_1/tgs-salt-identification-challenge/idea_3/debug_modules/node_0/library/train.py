import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Import components from the provided library files
from library.utils import set_seed, calculate_map_score
from library.losses import BCEDiceLoss, LovaszLoss
from library.model import HyperColumnUNet
from library.dataset import SaltDataset, get_transforms


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Calculates loss and mAP after cropping predictions back to original 101x101 size.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_truths = []

    # Padding constants used in SaltDataset (101 -> 128)
    # Target 128, Orig 101 => Diff 27. Top=13, Bottom=14, Left=13, Right=14.
    pad_top = 13
    pad_left = 13
    orig_h = 101
    orig_w = 101

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item() * images.size(0)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs).cpu().numpy()
            true_masks = masks.cpu().numpy()

            # Crop back to 101x101 for metric calculation
            for i in range(probs.shape[0]):
                # probs shape: (B, 1, 128, 128)
                pred_crop = probs[
                    i, 0, pad_top : pad_top + orig_h, pad_left : pad_left + orig_w
                ]
                true_crop = true_masks[
                    i, 0, pad_top : pad_top + orig_h, pad_left : pad_left + orig_w
                ]

                all_preds.append(pred_crop)
                all_truths.append(true_crop)

    avg_loss = running_loss / len(loader.dataset)

    # Binarize predictions at 0.5 threshold for mAP calculation
    # The metric function sweeps thresholds (0.5 to 0.95), but we need binary inputs or
    # specific format. calculate_map_score in utils.py takes raw masks and calculates IoU.
    # Based on utils.py: "pred_mask = pred_mask.astype(bool)".
    # So we pass the probability map thresholded at 0.5 as the "predicted object".
    bin_preds = [p > 0.5 for p in all_preds]
    bin_truths = [t > 0.5 for t in all_truths]

    map_score = calculate_map_score(bin_preds, bin_truths)

    return avg_loss, map_score


def run_fold(
    train_metadata="./metadata/train.csv",
    val_metadata="./metadata/val.csv",
    output_dir="./working/idea_3",
    epochs=50,
    batch_size=32,
    lr=1e-3,
    device=None,
    num_workers=4,
):
    """
    Executes the training pipeline for a single split (fold).
    Implements the two-stage training strategy:
    1. BCE + Dice Loss for convergence.
    2. Lovasz-Softmax Loss for fine-tuning.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(42)
    os.makedirs(output_dir, exist_ok=True)
    model_save_path = os.path.join(output_dir, "best_model.pth")

    # --- Data Loading ---
    train_transform = get_transforms(mode="train")
    val_transform = get_transforms(mode="val")

    # SaltDataset handles caching automatically in ./working/idea_3/ via default args
    train_dataset = SaltDataset(
        metadata_csv=train_metadata,
        transform=train_transform,
        mode="train",
        cache_dir=output_dir,
    )
    val_dataset = SaltDataset(
        metadata_csv=val_metadata,
        transform=val_transform,
        mode="val",
        cache_dir=output_dir,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # --- Model Setup ---
    # Input channels = 2 (Grayscale Image + Depth Channel)
    model = HyperColumnUNet(input_channels=2, num_classes=1, base_filters=32)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Scheduler monitors mAP (maximize)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, verbose=False
    )

    # --- Losses ---
    criterion_stage1 = BCEDiceLoss()
    criterion_stage2 = LovaszLoss()

    best_map = 0.0

    print(f"Starting training on {device} for {epochs} epochs...")
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    for epoch in range(1, epochs + 1):
        # Two-stage optimization logic
        # Switch to Lovasz loss after 50% of epochs
        use_lovasz = epoch > (epochs // 2)

        if use_lovasz:
            criterion = criterion_stage2
            loss_name = "Lovasz"
        else:
            criterion = criterion_stage1
            loss_name = "BCE+Dice"

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_map = validate(model, val_loader, criterion, device)

        # Update Scheduler
        scheduler.step(val_map)

        # Checkpoint
        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), model_save_path)
            saved_msg = " [Saved Best]"
        else:
            saved_msg = ""

        print(
            f"Epoch {epoch}/{epochs} | Loss ({loss_name}): Train={train_loss:.6f}, Val={val_loss:.6f} | mAP: {val_map:.15f}{saved_msg}"
        )

    print(f"Training finished. Best mAP: {best_map:.15f}")
    return model
