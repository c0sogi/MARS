import os
import random
import numpy as np
import pandas as pd
import cv2
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.model import HRNetSegmentation
from library.dataset import UWMadisonDataset
from library.losses import BCETverskyLoss
from library.utils import calculate_dice, rle_encode


def set_seed(seed=Config.SEED):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_transforms(phase):
    """
    Returns the Albumentations transforms for the specified phase.
    """
    if phase == "train":
        return A.Compose(
            [
                A.PadIfNeeded(
                    min_height=Config.TRAIN_CROP_SIZE[0],
                    min_width=Config.TRAIN_CROP_SIZE[1],
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                A.RandomCrop(
                    height=Config.TRAIN_CROP_SIZE[0], width=Config.TRAIN_CROP_SIZE[1]
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=30, p=0.5),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
                A.GridDistortion(p=0.2),
                ToTensorV2(transpose_mask=True),
            ]
        )
    else:
        # For Val/Test, we just convert to tensor.
        # Resizing/Padding is handled in the dataset or validation loop.
        return A.Compose(
            [
                ToTensorV2(transpose_mask=True),
            ]
        )


def keep_largest_connected_component(mask):
    """
    Keeps only the largest connected component for each class slice-wise.
    """
    mask = mask.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    if num_labels < 2:
        return mask

    # stats[:, 4] is area. Index 0 is background.
    max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    new_mask = np.zeros_like(mask)
    new_mask[labels == max_label] = 1
    return new_mask


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    """
    Executes one epoch of training.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        batch_size = images.size(0)

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Validates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dice_scores = []
    dataset_size = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            batch_size = images.size(0)

            # Handle padding for HRNet (needs dimensions divisible by 32)
            h, w = images.shape[2], images.shape[3]
            pad_h = (32 - h % 32) % 32
            pad_w = (32 - w % 32) % 32

            if pad_h > 0 or pad_w > 0:
                images = F.pad(images, (0, pad_w, 0, pad_h), mode="constant", value=0)

            with autocast():
                outputs = model(images)

                # Crop back to original size
                outputs = outputs[:, :, :h, :w]
                loss = criterion(outputs, masks)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Calculate Dice
            preds = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
            targets = masks.cpu().numpy()

            for i in range(preds.shape[0]):
                dice_scores.append(calculate_dice(targets[i], preds[i]))

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    epoch_dice = np.mean(dice_scores) if dice_scores else 0.0

    return epoch_loss, epoch_dice


def run_inference(model, device):
    """
    Generates predictions for the test set and saves to submission.csv.
    """
    print("Starting inference...")
    model.eval()

    df_test = pd.read_csv(Config.TEST_METADATA_PATH)
    test_dataset = UWMadisonDataset(
        df_test, phase="test", transform=get_transforms("test"), load_cached_data=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=Config.NUM_WORKERS
    )

    results = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            orig_shape = batch["orig_shape"].numpy()[0]  # (h, w)
            img_id = batch["id"][0]

            # Pad for HRNet
            h, w = images.shape[2], images.shape[3]
            pad_h = (32 - h % 32) % 32
            pad_w = (32 - w % 32) % 32

            if pad_h > 0 or pad_w > 0:
                images = F.pad(images, (0, pad_w, 0, pad_h), mode="constant", value=0)

            with autocast():
                outputs = model(images)

            # Crop padding
            outputs = outputs[:, :, :h, :w]
            probs = torch.sigmoid(outputs).cpu().numpy()[0]  # (C, H, W)

            final_masks = []
            for c in range(Config.NUM_CLASSES):
                prob_map = probs[c]

                # Resize to original resolution if needed
                if prob_map.shape != tuple(orig_shape):
                    prob_map = cv2.resize(
                        prob_map,
                        (orig_shape[1], orig_shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )

                mask = (prob_map > 0.5).astype(np.uint8)

                # Post-processing
                mask = keep_largest_connected_component(mask)

                rle = rle_encode(mask)
                final_masks.append(rle)

            results.append(
                {"id": img_id, "class": "large_bowel", "predicted": final_masks[0]}
            )
            results.append(
                {"id": img_id, "class": "small_bowel", "predicted": final_masks[1]}
            )
            results.append(
                {"id": img_id, "class": "stomach", "predicted": final_masks[2]}
            )

    submission_df = pd.DataFrame(results)
    submission_df = submission_df[["id", "class", "predicted"]]
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_training(epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE):
    """
    Main driver function to train the model and run inference.
    """
    set_seed()
    device = Config.DEVICE
    print(f"Using device: {device}")

    # Load Metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)

    # Datasets
    train_dataset = UWMadisonDataset(
        df_train,
        phase="train",
        transform=get_transforms("train"),
        load_cached_data=True,
    )
    val_dataset = UWMadisonDataset(
        df_val, phase="val", transform=get_transforms("val"), load_cached_data=True
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    # Val loader must have batch_size=1 due to variable image sizes
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Model
    model = HRNetSegmentation(
        num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED
    ).to(device)

    # Optimizer & Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=Config.ETA_MIN
    )
    criterion = BCETverskyLoss(
        alpha=Config.TVERSKY_ALPHA,
        beta=Config.TVERSKY_BETA,
        smooth=Config.TVERSKY_SMOOTH,
        bce_weight=Config.WEIGHT_BCE,
        tversky_weight=Config.WEIGHT_TVERSKY,
    )
    scaler = GradScaler()

    best_dice = 0.0

    print("Starting training...")
    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Dice: {val_dice:.6f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"New best model saved with Dice: {val_dice:.6f}")

    print("Training complete.")

    # Run Inference
    # Reload best model
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    run_inference(model, device)
