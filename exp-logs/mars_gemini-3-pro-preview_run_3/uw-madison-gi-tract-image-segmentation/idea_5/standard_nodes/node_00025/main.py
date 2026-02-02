import os
import sys
import pandas as pd
import numpy as np
import torch
import gc
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from scipy.stats import pearsonr

# Import from library
from library.config import CFG
from library.dataset import UWMGIDataset, get_transforms, process_25d_dataframe
from library.model import build_model
from library.train import train as run_training
from library.inference import predict_and_submit
from library.utils import compute_metrics, rle_encode, rle_decode, compute_dice


def failure_analysis(model, df_val, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between error magnitude and input features.
    """
    print("\n=== Running Failure Analysis ===")

    # Create dataset/loader for validation
    val_dataset = UWMGIDataset(
        df_val,
        label=True,  # We need masks for per-slice analysis
        transforms=get_transforms(data="valid"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.valid_batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model.eval()

    # Store metrics and features
    errors = []
    feat_slice_norm = []
    feat_height = []
    feat_width = []

    # Get max slice per case/day for normalization
    df_val["slice_int"] = df_val["slice"].astype(int)
    max_slices = df_val.groupby(["case", "day"])["slice_int"].transform("max")
    df_val["slice_norm"] = df_val["slice_int"] / max_slices

    # Iterate
    idx_counter = 0
    with torch.no_grad():
        for images, masks, ids in val_loader:
            images = images.to(device, dtype=torch.float)
            masks = masks.to(device, dtype=torch.float)

            with autocast(enabled=CFG.mixed_precision):
                logits = model(images)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                probs = torch.sigmoid(logits)
                preds = (probs > CFG.mask_threshold).float()

            # Compute per-sample Dice (2D)
            # Shapes: (B, C, H, W)
            # We compute mean dice across classes for the slice
            batch_size = images.size(0)

            preds_np = preds.cpu().numpy()
            masks_np = masks.cpu().numpy()

            for i in range(batch_size):
                # Get metadata features for this sample
                row = df_val.iloc[idx_counter]
                idx_counter += 1

                # Compute Dice for this slice (average over 3 classes)
                dice_sum = 0
                for c in range(3):
                    d = compute_dice(masks_np[i, c], preds_np[i, c])
                    dice_sum += d
                mean_slice_dice = dice_sum / 3.0

                # Error magnitude
                error = 1.0 - mean_slice_dice

                errors.append(error)
                feat_slice_norm.append(row["slice_norm"])
                feat_height.append(row["height"])
                feat_width.append(row["width"])

    # Calculate correlations
    if len(errors) > 1:
        corr_slice, _ = pearsonr(errors, feat_slice_norm)
        corr_h, _ = pearsonr(errors, feat_height)
        corr_w, _ = pearsonr(errors, feat_width)

        print(f"Correlation (Error vs Normalized Slice Position): {corr_slice:.4f}")
        print(f"Correlation (Error vs Image Height): {corr_h:.4f}")
        print(f"Correlation (Error vs Image Width): {corr_w:.4f}")
    else:
        print("Insufficient data for correlation analysis.")


def main():
    # 1. Configuration for Fast Baseline
    # Adjust epochs to fit within time limit (A100 is fast, 5 epochs is safe and sufficient)
    CFG.epochs = 12
    CFG.debug = False  # Use full dataset for best results

    # 2. Run Training
    # This saves the best model to ./working/idea_5/checkpoints/best_model.pth
    print("Starting Training Pipeline...")
    run_training()

    # 3. Validation Assessment
    print("\n=== Validation Assessment ===")
    device = CFG.device

    # Load metadata
    df_val = pd.read_csv(CFG.val_csv)
    df_val = process_25d_dataframe(df_val, split_name="val", load_cached_data=True)

    # Load Best Model
    model = build_model()
    checkpoint_path = os.path.join(CFG.checkpoint_dir, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        print("Error: Checkpoint not found.")
        return

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Generate Predictions for Validation Set (Full 3D Metric Calculation)
    # We reuse the logic from valid_one_epoch but need to ensure we cover the whole set
    # Create loader
    val_dataset_inf = UWMGIDataset(
        df_val, label=False, transforms=get_transforms(data="valid")
    )
    val_loader_inf = DataLoader(
        val_dataset_inf,
        batch_size=CFG.valid_batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    pred_rows = []
    classes = ["large_bowel", "small_bowel", "stomach"]

    print("Generating validation predictions...")
    with torch.no_grad():
        for images, ids in val_loader_inf:
            images = images.to(device, dtype=torch.float)

            with autocast(enabled=CFG.mixed_precision):
                logits = model(images)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                probs = torch.sigmoid(logits)
                preds = (probs > CFG.mask_threshold).float()

            preds_np = preds.cpu().numpy().astype(np.uint8)

            for i in range(len(ids)):
                sample_id = ids[i]
                for c_idx, c_name in enumerate(classes):
                    rle = rle_encode(preds_np[i, c_idx])
                    pred_rows.append(
                        {"id": sample_id, "class": c_name, "predicted": rle}
                    )

    df_pred = pd.DataFrame(pred_rows)

    # Compute Final Metric
    metrics = compute_metrics(df_pred, df_val)
    final_score = metrics["score"]

    # REQUIRED PRINT FORMAT
    print(f"Final Validation Metric: {final_score}")

    # 4. Failure Analysis
    failure_analysis(model, df_val, device)

    # Clean up memory
    del model, val_loader_inf, df_pred
    gc.collect()
    torch.cuda.empty_cache()

    # 5. Submission
    THRESHOLD = 0.534075
    if final_score > THRESHOLD:
        print(
            f"\nMetric ({final_score}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        predict_and_submit(load_cached_data=True)
    else:
        print(
            f"\nMetric ({final_score}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
