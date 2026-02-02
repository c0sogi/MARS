import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from provided library files
from library.utils import set_seed
from library.dataset import load_and_cache_data, SaltDataset, get_transforms
from library.model import DepthAwareLinkNet34
from library.losses import CombinedLoss
from library.trainer import train_one_epoch, validate, generate_submission

# Constants
IDEA_DIR = "./working/idea_12"
EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_failure_analysis(model, val_loader, raw_val_depths, best_threshold):
    """
    Performs failure analysis on the validation set by correlating
    performance with depth and salt coverage.
    """
    print("Performing failure analysis...")
    model.eval()

    all_preds = []
    all_masks = []

    # Crop indices for 128 -> 101 conversion
    IMG_SIZE = 128
    ORIG_SIZE = 101
    start_idx = (IMG_SIZE - ORIG_SIZE) // 2
    end_idx = start_idx + ORIG_SIZE

    with torch.no_grad():
        for images, masks, depths in val_loader:
            images = images.to(DEVICE)
            depths = depths.to(DEVICE)

            # Forward pass
            logits = model(images, depths)
            probs = torch.sigmoid(logits)

            # Crop to original size
            probs = probs[:, :, start_idx:end_idx, start_idx:end_idx]
            masks = masks[:, :, start_idx:end_idx, start_idx:end_idx]

            all_preds.append(probs.cpu().numpy())
            all_masks.append(masks.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_masks = np.concatenate(all_masks)

    # Calculate per-image IoU/mAP
    iou_thresholds = np.arange(0.5, 0.96, 0.05)
    scores = []
    salt_coverages = []

    # Binarize predictions using the optimal threshold
    preds_bin = (all_preds > best_threshold).astype(np.uint8)
    masks_bin = (all_masks > 0.5).astype(np.uint8)

    for i in range(len(all_preds)):
        p = preds_bin[i, 0]
        m = masks_bin[i, 0]

        intersection = np.sum(p & m)
        union = np.sum(p | m)

        if union == 0:
            iou = 1.0
        else:
            iou = intersection / union

        # mAP calculation for this single image
        matches = iou > iou_thresholds
        score = np.mean(matches)
        scores.append(score)

        # Calculate salt coverage (ground truth)
        salt_coverages.append(np.sum(m) / m.size)

    # Create DataFrame for analysis
    # Note: val_loader is sequential (shuffle=False), so it aligns with raw_val_depths
    df_analysis = pd.DataFrame(
        {"score": scores, "depth": raw_val_depths, "salt_coverage": salt_coverages}
    )

    # Calculate correlations
    corr_depth = df_analysis["score"].corr(df_analysis["depth"])
    corr_coverage = df_analysis["score"].corr(df_analysis["salt_coverage"])

    print(f"Correlation (Score vs Depth): {corr_depth}")
    print(f"Correlation (Score vs Salt Coverage): {corr_coverage}")


def main():
    # 1. Setup
    set_seed(42)
    os.makedirs(IDEA_DIR, exist_ok=True)

    # 2. Load Data
    data = load_and_cache_data(load_cached=True)

    # Calculate depth stats for normalization
    all_depths = np.concatenate([data["train_depths"], data["val_depths"]])
    d_mean = all_depths.mean()
    d_std = all_depths.std()

    # 3. Initialize Datasets & Loaders
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 4. Initialize Model
    model = DepthAwareLinkNet34().to(DEVICE)
    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_map = 0.0
    best_thresh = 0.5

    # 5. Training Loop
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_map, thresh = validate(model, val_loader, DEVICE)
        scheduler.step()

        if val_map > best_map:
            best_map = val_map
            best_thresh = thresh
            torch.save(model.state_dict(), os.path.join(IDEA_DIR, "best_model.pth"))

    # 6. Final Metric
    print(f"Final Validation Metric: {best_map}")

    # 7. Failure Analysis
    # Load best model
    model.load_state_dict(torch.load(os.path.join(IDEA_DIR, "best_model.pth")))
    run_failure_analysis(model, val_loader, data["val_depths"], best_thresh)

    # 8. Submission
    TARGET_METRIC = 0.7985
    if best_map > TARGET_METRIC:
        generate_submission(best_thresh, d_mean, d_std)


if __name__ == "__main__":
    main()
