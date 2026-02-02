import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import from provided library files
from library.utils import set_seed, calculate_iou_map
from library.dataset import get_dataloaders, TARGET_H, TARGET_W, ORIG_H, ORIG_W
from library.model import DeepResUNet
from library.training import Trainer, CHECKPOINT_DIR


def main():
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration
    # -------------------------------------------------------------------------
    set_seed(42)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # 2. Training
    # -------------------------------------------------------------------------
    # Load data with caching enabled
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=32, load_cached_data=True, num_workers=4
    )

    # Initialize Model and Trainer
    model = DeepResUNet(in_channels=1, out_channels=1)
    trainer = Trainer(model, train_loader, val_loader, test_loader)

    # Execute Training
    # We run for 150 epochs to complete the 3 cycles (50 epochs each)
    # required by the curriculum strategy.
    trainer.run(epochs=150)

    # -------------------------------------------------------------------------
    # 3. Validation Assessment
    # -------------------------------------------------------------------------
    print("Performing Validation Assessment...")

    # Paths to the best checkpoints saved during training
    path_c2 = os.path.join(CHECKPOINT_DIR, "best_cycle_2.pth")
    path_c3 = os.path.join(CHECKPOINT_DIR, "best_cycle_3.pth")

    # Initialize models for inference
    model_c2 = DeepResUNet().to(DEVICE)
    model_c3 = DeepResUNet().to(DEVICE)

    c2_exists = os.path.exists(path_c2)
    c3_exists = os.path.exists(path_c3)

    if c2_exists:
        model_c2.load_state_dict(torch.load(path_c2, map_location=DEVICE))
    if c3_exists:
        model_c3.load_state_dict(torch.load(path_c3, map_location=DEVICE))

    model_c2.eval()
    model_c3.eval()

    # Quality-Gated Ensembling Logic
    map_c2 = trainer.best_map_c2
    map_c3 = trainer.best_map_c3

    # If checkpoints weren't saved (e.g. if training crashed or didn't improve), handle gracefully
    if not c2_exists:
        map_c2 = 0.0
    if not c3_exists:
        map_c3 = 0.0

    diff = abs(map_c2 - map_c3)
    use_ensemble = (diff < 0.005) and c2_exists and c3_exists

    # Determine best single model
    best_single_model = model_c2 if map_c2 >= map_c3 else model_c3
    if not c2_exists and not c3_exists:
        # Fallback to current model state if no checkpoints
        best_single_model = trainer.model

    # Inference Loop on Validation Set
    all_preds = []
    all_masks = []
    all_depths = []

    # Cropping indices
    start_h = (TARGET_H - ORIG_H) // 2
    start_w = (TARGET_W - ORIG_W) // 2

    with torch.no_grad():
        for images, masks, depths, _ in val_loader:
            images = images.to(DEVICE)
            depths_gpu = depths.to(DEVICE)

            # TTA Helper Function (Horizontal Flip)
            def predict_tta(m, x, z):
                pred = torch.sigmoid(m(x, z))
                x_flip = torch.flip(x, [3])
                pred_flip = torch.sigmoid(m(x_flip, z))
                pred_flip_back = torch.flip(pred_flip, [3])
                return (pred + pred_flip_back) / 2.0

            # Generate Predictions
            if use_ensemble:
                p2 = predict_tta(model_c2, images, depths_gpu)
                p3 = predict_tta(model_c3, images, depths_gpu)
                avg_pred = (p2 + p3) / 2.0
            else:
                avg_pred = predict_tta(best_single_model, images, depths_gpu)

            # Crop to original 101x101 resolution
            avg_pred = avg_pred[
                :, 0, start_h : start_h + ORIG_H, start_w : start_w + ORIG_W
            ]

            # Retrieve masks and crop (masks are usually padded in dataset)
            masks_cropped = masks[
                :, 0, start_h : start_h + ORIG_H, start_w : start_w + ORIG_W
            ]

            all_preds.append(avg_pred.cpu().numpy())
            all_masks.append(masks_cropped.numpy())
            all_depths.extend(depths.numpy().flatten())

    # Concatenate results
    all_preds = np.concatenate(all_preds, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    all_depths = np.array(all_depths)

    # Binarize for Metric Calculation
    binary_preds = (all_preds > 0.5).astype(np.uint8)
    binary_masks = (all_masks > 0.5).astype(np.uint8)

    # Compute Final Metric
    final_metric = calculate_iou_map(binary_preds, binary_masks, verbose=False)
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 4. Failure Analysis
    # -------------------------------------------------------------------------
    print("\nRunning Failure Analysis...")

    # Calculate mAP per image to correlate with metadata
    # We use the library function on individual samples
    errors = []
    coverages = []

    for i in range(len(binary_preds)):
        # Slice to keep dimensions (1, H, W)
        p = binary_preds[i : i + 1]
        m = binary_masks[i : i + 1]

        # Calculate score for this image
        score = calculate_iou_map(p, m)
        errors.append(1.0 - score)

        # Calculate coverage (proportion of salt pixels)
        cov = np.mean(m)
        coverages.append(cov)

    errors = np.array(errors)
    coverages = np.array(coverages)

    # Calculate Correlations
    # Handle constant arrays to avoid warnings
    if np.std(errors) > 0 and np.std(all_depths) > 0:
        corr_depth, _ = pearsonr(errors, all_depths)
    else:
        corr_depth = 0.0

    if np.std(errors) > 0 and np.std(coverages) > 0:
        corr_cov, _ = pearsonr(errors, coverages)
    else:
        corr_cov = 0.0

    print(f"Failure Analysis - Correlation (Error vs Depth): {corr_depth:.4f}")
    print(f"Failure Analysis - Correlation (Error vs Salt Coverage): {corr_cov:.4f}")

    # -------------------------------------------------------------------------
    # 5. Submission Generation
    # -------------------------------------------------------------------------
    if final_metric > 0.833:
        # The trainer class has the logic to generate submission using the same
        # Quality-Gated Ensembling strategy.
        trainer.predict_test_set()
    else:
        print(
            f"Validation metric {final_metric} is below threshold 0.833. Skipping submission."
        )


if __name__ == "__main__":
    main()
