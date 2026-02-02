import os
import numpy as np
import pandas as pd
import torch
from library import config, utils, dataset, model, train, predict, losses


def main():
    # 1. Setup
    utils.set_seed(config.SEED)
    device = config.DEVICE
    print(f"Orchestration started on device: {device}")

    # 2. Training
    # We use 30 epochs to strike a balance between a fast baseline and sufficient convergence
    # to meet the high metric threshold.
    print("\n=== Starting Training Phase ===")
    depth_mean, depth_std = train.run_training(
        epochs=30, batch_size=config.BATCH_SIZE, debug=False, device=device
    )

    # 3. Validation & Metric Calculation
    print("\n=== Starting Validation Phase ===")
    # Reload loaders to ensure we are evaluating the exact hold-out set
    _, val_loader, _, _ = dataset.get_train_val_loaders(load_cached_data=True)

    # Load the best model checkpoint
    net = model.WideLinkNet34().to(device)
    if not os.path.exists(config.CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at {config.CHECKPOINT_PATH}")

    net.load_state_dict(torch.load(config.CHECKPOINT_PATH, map_location=device))

    # Run validation to get metrics and predictions
    criterion = losses.SaltNetLoss()
    val_loss, val_map, val_preds, val_masks = model.validate(
        net, val_loader, criterion, device
    )

    # Print the required metric
    print(f"Final Validation Metric: {val_map}")

    # 4. Failure Analysis
    print("\n=== Starting Failure Analysis ===")

    # Convert tensors to numpy for analysis
    # val_preds: (N, 1, H, W) probabilities
    # val_masks: (N, 1, H, W) binary ground truth
    preds_np = val_preds.numpy()
    masks_np = val_masks.numpy()

    # Flatten spatial dimensions for IoU calculation: (N, H*W)
    preds_flat = preds_np.reshape(preds_np.shape[0], -1)
    masks_flat = masks_np.reshape(masks_np.shape[0], -1)

    # Calculate AP per image
    # We use a fixed pixel threshold of 0.5 for the analysis to determine "hits"
    pixel_threshold = 0.5
    preds_bin = (preds_flat > pixel_threshold).astype(np.uint8)
    masks_bin = (masks_flat > 0.5).astype(np.uint8)

    intersection = (preds_bin & masks_bin).sum(axis=1)
    union = (preds_bin | masks_bin).sum(axis=1)

    # Calculate IoU
    iou = np.ones(preds_bin.shape[0], dtype=np.float32)
    non_empty = union > 0
    iou[non_empty] = intersection[non_empty] / union[non_empty]

    # Calculate Average Precision over thresholds 0.5 to 0.95
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    # matches shape: (N, 10)
    matches = iou[:, None] > iou_thresholds[None, :]
    image_aps = matches.mean(axis=1)

    # Error is 1 - AP
    errors = 1.0 - image_aps

    # Retrieve Metadata
    # val_loader.dataset.depths is aligned with the loader output (shuffle=False)
    val_depths = val_loader.dataset.depths

    # Calculate Salt Coverage from the masks used in validation
    # This ensures we use the cropped/processed mask area correctly
    img_area = masks_flat.shape[1]
    val_coverage = masks_flat.sum(axis=1) / img_area

    # Compute Correlations
    if len(errors) == len(val_depths):
        df_analysis = pd.DataFrame(
            {"error": errors, "depth": val_depths, "coverage": val_coverage}
        )

        corr_depth = df_analysis["error"].corr(df_analysis["depth"])
        corr_cov = df_analysis["error"].corr(df_analysis["coverage"])

        print(f"Correlation (Error vs Depth): {corr_depth:.10f}")
        print(f"Correlation (Error vs Salt Coverage): {corr_cov:.10f}")
    else:
        print("Warning: Metadata length mismatch. Skipping correlation analysis.")

    # 5. Submission
    target_threshold = 0.7916666666666666
    print(f"\nTarget Threshold: {target_threshold}")

    if val_map > target_threshold:
        print("Validation metric exceeds threshold. Proceeding to submission...")
        # Run inference pipeline
        # Pass depth stats to ensure consistent normalization
        predict.predict(depth_mean, depth_std)
    else:
        print("Validation metric does not exceed threshold. Submission skipped.")


if __name__ == "__main__":
    main()
