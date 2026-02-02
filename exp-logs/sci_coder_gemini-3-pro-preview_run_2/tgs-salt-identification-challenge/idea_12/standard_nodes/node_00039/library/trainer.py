import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from library.utils import set_seed, rle_encode, calculate_iou_map
from library.losses import CombinedLoss
from library.model import DepthAwareLinkNet34
from library.dataset import load_and_cache_data, SaltDataset, get_transforms

# Constants
IDEA_DIR = "./working/idea_12"
IMG_SIZE = 128
ORIG_SIZE = 101
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Executes one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for images, masks, depths in loader:
        images = images.to(device)
        masks = masks.to(device)
        depths = depths.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images, depths)

        # Compute loss
        loss = criterion(outputs, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    Performs linear search for the best IoU threshold.
    """
    model.eval()
    all_preds = []
    all_masks = []

    # Calculate crop indices to revert 128x128 padding to 101x101
    # Padding was centered (or default albumentations behavior).
    # (128 - 101) // 2 = 13.
    start_idx = (IMG_SIZE - ORIG_SIZE) // 2
    end_idx = start_idx + ORIG_SIZE

    with torch.no_grad():
        for images, masks, depths in loader:
            images = images.to(device)
            depths = depths.to(device)

            # Forward pass
            logits = model(images, depths)
            probs = torch.sigmoid(logits)

            # Crop back to original size
            probs_cropped = probs[:, :, start_idx:end_idx, start_idx:end_idx]

            # Move to CPU
            all_preds.append(probs_cropped.cpu().numpy())

            # Masks in loader are already transformed (padded). We need to crop them too
            # or rely on the fact that we should compare against original masks.
            # The loader returns padded masks. We crop them to ensure consistency.
            masks_cropped = masks[:, :, start_idx:end_idx, start_idx:end_idx]
            all_masks.append(masks_cropped.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_masks = np.concatenate(all_masks)

    # Linear search for best threshold
    best_threshold = 0.5
    best_score = -1.0

    # Search range: 0.3 to 0.7
    for t in np.arange(0.3, 0.75, 0.05):
        score = calculate_iou_map(all_preds, all_masks, threshold=t)
        if score > best_score:
            best_score = score
            best_threshold = t

    return best_score, best_threshold


def run_training(epochs=50, batch_size=32, lr=1e-4):
    """
    Main training routine.
    """
    set_seed(42)
    os.makedirs(IDEA_DIR, exist_ok=True)

    # Load Data
    data = load_and_cache_data(load_cached=True)

    # Calculate depth statistics from Train + Val
    all_depths = np.concatenate([data["train_depths"], data["val_depths"]])
    d_mean = all_depths.mean()
    d_std = all_depths.std()

    # Initialize Datasets
    train_dataset = SaltDataset(
        data["train_images"],
        data["train_masks"],
        data["train_depths"],
        transform=get_transforms("train"),
        depth_mean=d_mean,
        depth_std=d_std,
        training=True,
    )

    val_dataset = SaltDataset(
        data["val_images"],
        data["val_masks"],
        data["val_depths"],
        transform=get_transforms("val"),
        depth_mean=d_mean,
        depth_std=d_std,
        training=False,
    )

    # Initialize Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Initialize Model, Loss, Optimizer
    model = DepthAwareLinkNet34().to(DEVICE)
    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_map = 0.0
    best_thresh = 0.5
    patience = 10
    no_improve = 0

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_map, thresh = validate(model, val_loader, DEVICE)
        scheduler.step()

        # Print full precision as requested
        print(
            f"Epoch {epoch+1}/{epochs} | Loss: {train_loss} | Val mAP: {val_map} | Thresh: {thresh}"
        )

        if val_map > best_map:
            best_map = val_map
            best_thresh = thresh
            torch.save(model.state_dict(), os.path.join(IDEA_DIR, "best_model.pth"))
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print("Early stopping triggered.")
            break

    print(f"Best Validation mAP: {best_map} at Threshold: {best_thresh}")
    return best_thresh, d_mean, d_std


def generate_submission(best_threshold, d_mean, d_std):
    """
    Generates submission file using the best model and TTA.
    """
    print("Generating submission...")
    data = load_and_cache_data(load_cached=True)

    # Initialize Test Dataset
    # Note: SaltDataset automatically sets z=0 if masks is None, matching inference strategy.
    test_dataset = SaltDataset(
        data["test_images"],
        None,
        data["test_depths"],
        transform=get_transforms("test"),
        depth_mean=d_mean,
        depth_std=d_std,
        training=False,
    )

    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
    )

    # Load Model
    model = DepthAwareLinkNet34().to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(IDEA_DIR, "best_model.pth")))
    model.eval()

    rles = []
    ids = data["test_ids"]

    # Crop indices
    start_idx = (IMG_SIZE - ORIG_SIZE) // 2
    end_idx = start_idx + ORIG_SIZE

    with torch.no_grad():
        for images, depths in test_loader:
            images = images.to(DEVICE)
            depths = depths.to(DEVICE)

            # TTA: Original
            out1 = torch.sigmoid(model(images, depths))

            # TTA: Horizontal Flip
            images_flipped = torch.flip(images, dims=[3])
            out2 = torch.sigmoid(model(images_flipped, depths))
            out2 = torch.flip(out2, dims=[3])

            # Average
            preds = (out1 + out2) / 2.0

            # Crop
            preds = preds[:, 0, start_idx:end_idx, start_idx:end_idx]

            # Threshold and Encode
            preds_bin = (preds > best_threshold).cpu().numpy().astype(np.uint8)

            for p in preds_bin:
                rles.append(rle_encode(p))

    # Save Submission
    sub_df = pd.DataFrame({"id": ids, "rle_mask": rles})
    os.makedirs("submission", exist_ok=True)
    sub_df.to_csv("submission/submission.csv", index=False)
    print("Submission saved to submission/submission.csv")


def run_task():
    """
    Entry point for the task.
    """
    thresh, dm, ds = run_training(epochs=50)
    generate_submission(thresh, dm, ds)
