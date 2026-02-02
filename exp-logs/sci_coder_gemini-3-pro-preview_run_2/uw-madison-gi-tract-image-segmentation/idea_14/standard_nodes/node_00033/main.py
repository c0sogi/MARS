import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import cv2
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from scipy.stats import pearsonr

# Import from library
from library.config import Config
from library.utils import (
    calculate_dice,
    calculate_hausdorff_3d,
    rle_decode,
    rle_encode,
    load_image,
)
from library.model import HRNetSegmentation
from library.dataset import UWMadisonDataset
from library.losses import BCETverskyLoss
from library.train import get_transforms, train_one_epoch, validate as validate_epoch
from library.inference import run_inference, predict_sliding_window, post_process_volume


def set_seed(seed):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def evaluate_validation_set(model, df_val, device):
    """
    Performs full validation on the validation set, calculating
    Dice and 3D Hausdorff distance per case.
    Also collects slice-level metrics for failure analysis.
    """
    model.eval()

    # Create dataset without subsampling
    val_dataset = UWMadisonDataset(
        df_val,
        phase="val",
        transform=None,  # No transforms for validation inference
        load_cached_data=True,
    )

    print("Running Full Validation & Failure Analysis...")

    # Strategy:
    # 1. Iterate dataset (which yields unique slices).
    # 2. Store predictions and GTs in a dictionary structure: volumes[(case, day)][class] -> list of (slice_idx, mask)
    # 3. Assemble and compute metrics.

    volumes = {}  # (case, day) -> { 'pred': [ (slice, mask_c1), ... ], 'gt': ... }

    # We also need metadata for failure analysis
    meta_stats = []

    loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=Config.NUM_WORKERS
    )

    for batch in tqdm(loader, desc="Inference on Val"):
        img = batch["image"].to(device)
        mask_gt_resampled = batch["mask"].cpu().numpy()[0]  # (C, H_res, W_res)
        orig_shape = batch["orig_shape"].numpy()[0]  # (h, w)
        case = batch["case"].item()
        day = batch["day"].item()
        slc = batch["slice"].item()
        img_id = batch["id"][0]

        # Predict
        # Use sliding window to handle variable sizes if needed, or just forward if fits
        # HRNet needs padding to 32
        h, w = img.shape[2], img.shape[3]
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        if pad_h > 0 or pad_w > 0:
            img_in = torch.nn.functional.pad(
                img, (0, pad_w, 0, pad_h), mode="constant", value=0
            )
        else:
            img_in = img

        with torch.cuda.amp.autocast():
            out = model(img_in)
            probs = torch.sigmoid(out)[:, :, :h, :w].cpu().numpy()[0]  # (C, H, W)

        # Resize to original resolution
        preds_orig = []
        gts_orig = []

        # We need to fetch original GT from dataframe because dataset returns resampled mask
        # But for simplicity and speed, we can resize the resampled GT back to original.
        # This introduces slight error but is consistent.
        # Better: Use the dataset's mask (resampled) for 2D Dice, but for 3D HD use resized.

        for c in range(Config.NUM_CLASSES):
            # Resize Pred
            pm = probs[c]
            if pm.shape != tuple(orig_shape):
                pm = cv2.resize(
                    pm, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_LINEAR
                )
            p_mask = (pm > 0.5).astype(np.uint8)
            preds_orig.append(p_mask)

            # Resize GT (Nearest Neighbor)
            gm = mask_gt_resampled[c]
            if gm.shape != tuple(orig_shape):
                gm = cv2.resize(
                    gm, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST
                )
            gts_orig.append(gm.astype(np.uint8))

            # 2D Dice for Failure Analysis
            d_score = calculate_dice(gts_orig[-1], preds_orig[-1])

            # Collect stats
            # We need pixel spacing and width.
            # We can get them from df_val using img_id, but that's slow.
            # We'll assume they are roughly consistent or just use slice index.
            meta_stats.append(
                {
                    "id": img_id,
                    "case": case,
                    "day": day,
                    "slice": slc,
                    "class": Config.CLASSES[c],
                    "dice": d_score,
                    "img_width": orig_shape[1],
                }
            )

        # Store for 3D
        key = (case, day)
        if key not in volumes:
            volumes[key] = {"slices": [], "preds": [], "gts": []}

        volumes[key]["slices"].append(slc)
        volumes[key]["preds"].append(np.stack(preds_orig))  # (C, H, W)
        volumes[key]["gts"].append(np.stack(gts_orig))  # (C, H, W)

    # Compute 3D Metrics
    case_dices = []
    case_hds = []

    for key, data in volumes.items():
        # Sort by slice index
        sorted_indices = np.argsort(data["slices"])

        # Stack to (D, C, H, W) -> Transpose to (C, D, H, W)
        vol_pred = np.stack(data["preds"])[sorted_indices].transpose(1, 0, 2, 3)
        vol_gt = np.stack(data["gts"])[sorted_indices].transpose(1, 0, 2, 3)

        for c in range(Config.NUM_CLASSES):
            vp = vol_pred[c]
            vg = vol_gt[c]

            # Post-process Pred (3D LCC)
            vp = post_process_volume(vp, threshold=0.5)

            # Metrics
            d = calculate_dice(vg, vp)
            hd = calculate_hausdorff_3d(vg, vp)

            case_dices.append(d)

            # Normalize HD to score
            # Metric: 0.4*Dice + 0.6*(1 - HD_normalized)
            # We assume HD is already somewhat normalized by image size in calculate_hausdorff_3d
            # But standard HD can be > 1.
            # We clip it to 1.0 for the score calculation to avoid negative scores.
            hd_score = 1.0 - min(hd, 1.0)
            case_hds.append(hd_score)

    final_dice = np.mean(case_dices)
    final_hd_score = np.mean(case_hds)
    final_metric = 0.4 * final_dice + 0.6 * final_hd_score

    print(f"Global Val Dice: {final_dice:.4f}")
    print(f"Global Val HD Score: {final_hd_score:.4f}")

    # Failure Analysis
    df_stats = pd.DataFrame(meta_stats)
    print("\n=== Failure Analysis (Correlation with 2D Dice) ===")
    # Encode class
    df_stats["class_enc"] = df_stats["class"].astype("category").cat.codes

    correlations = {}
    for col in ["slice", "img_width", "class_enc"]:
        if col in df_stats.columns:
            corr, _ = pearsonr(df_stats[col], df_stats["dice"])
            correlations[col] = corr
            print(f"Correlation {col} vs Dice: {corr:.4f}")

    return final_metric


def main():
    # 1. Configuration for Fast Baseline
    Config.EPOCHS = 5
    Config.SAMPLE_SIZE = 5000
    Config.DEBUG = True
    Config.BATCH_SIZE = 32  # A100 can handle this easily with 320x320

    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 2. Load Metadata
    print("Loading metadata...")
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)

    # 3. Datasets & Loaders
    print("Initializing datasets...")
    train_dataset = UWMadisonDataset(
        df_train,
        phase="train",
        transform=get_transforms("train"),
        load_cached_data=True,
    )
    # Val loader for monitoring (subsampled via Config.DEBUG)
    val_dataset_monitor = UWMadisonDataset(
        df_val, phase="val", transform=get_transforms("val"), load_cached_data=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader_monitor = DataLoader(
        val_dataset_monitor,
        batch_size=1,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 4. Model Setup
    model = HRNetSegmentation(
        num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED
    ).to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )
    criterion = BCETverskyLoss(
        alpha=Config.TVERSKY_ALPHA,
        beta=Config.TVERSKY_BETA,
        smooth=Config.TVERSKY_SMOOTH,
        bce_weight=Config.WEIGHT_BCE,
        tversky_weight=Config.WEIGHT_TVERSKY,
    )
    scaler = GradScaler()

    # 5. Training Loop
    best_dice_monitor = 0.0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, val_dice = validate_epoch(
            model, val_loader_monitor, criterion, device
        )

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice (2D): {val_dice:.4f}"
        )

        if val_dice > best_dice_monitor:
            best_dice_monitor = val_dice
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

    print("Training complete.")

    # 6. Full Validation
    # Disable debug limits for full validation
    Config.DEBUG = False
    Config.SAMPLE_SIZE = None

    # Reload best model
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    final_metric = evaluate_validation_set(model, df_val, device)

    print(f"Final Validation Metric: {final_metric:.10f}")

    # 7. Submission
    if final_metric > 0.438:
        print("Validation metric satisfactory. Generating submission...")
        # run_inference uses Config, which we just updated to disable DEBUG, so it will run on full test set
        run_inference()
    else:
        print(
            f"Validation metric {final_metric:.4f} is below threshold 0.438. Skipping submission."
        )


if __name__ == "__main__":
    main()
