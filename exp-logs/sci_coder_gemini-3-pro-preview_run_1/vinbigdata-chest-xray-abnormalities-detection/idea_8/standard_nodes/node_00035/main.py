import sys
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset

# Import from library
from library.config import Config
from library.utils import seed_everything, calculate_iou
from library.dataset import ChestXRayDataset, get_train_transforms, get_val_transforms
from library.model import DetModel
from library.engine import Engine


def failure_analysis(engine, val_loader):
    """
    Performs failure analysis on the validation set.
    Calculates correlations between error magnitude (1 - Recall) and input features.
    """
    print("\nStarting Failure Analysis...")
    engine.model.eval()
    device = engine.device

    # Access raw data for Ground Truth lookup
    # val_loader.dataset might be a Subset or the original Dataset.
    # If it's a Subset, we access .dataset. If it's the Dataset, we access .data directly.
    if isinstance(val_loader.dataset, Subset):
        raw_data = val_loader.dataset.dataset.data
    else:
        raw_data = val_loader.dataset.data

    gt_data_map = {item["image_id"]: item for item in raw_data}

    # Lists to store metrics and features
    error_magnitudes = []
    feat_area = []
    feat_num_objs = []
    feat_aspect_ratio = []

    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device)
            img_ids = batch["image_id"]
            orig_shapes = batch["original_shape"].numpy()

            outputs = engine.model(imgs)

            # Decode predictions
            dets = engine._decode_detections(
                outputs["hm"], outputs["wh"], outputs["reg"]
            )
            dets = dets.cpu().numpy()

            for i, img_id in enumerate(img_ids):
                # Retrieve Ground Truth
                if img_id not in gt_data_map:
                    continue

                gt_item = gt_data_map[img_id]
                # Filter out "No finding" (Class 14) for geometric analysis
                valid_mask = gt_item["labels"] != Config.NO_FINDING_CLASS_ID
                gt_boxes = gt_item["boxes"][valid_mask]

                # Retrieve Predictions
                pred_boxes = dets[i]
                # Filter by a reasonable confidence for analysis (e.g., > 0.1) to reduce noise
                pred_boxes = pred_boxes[pred_boxes[:, 4] > 0.1]

                # Rescale Predictions to Original Image Coordinates for fair comparison
                orig_h, orig_w = orig_shapes[i]
                scale_feat = 4.0  # Network stride

                # Scale from Feature Map to Input Size
                boxes_input = pred_boxes[:, :4] * scale_feat

                # Scale from Input Size to Original Size
                sx = orig_w / Config.IMAGE_SIZE
                sy = orig_h / Config.IMAGE_SIZE

                boxes_orig = boxes_input.copy()
                boxes_orig[:, 0] *= sx
                boxes_orig[:, 2] *= sx
                boxes_orig[:, 1] *= sy
                boxes_orig[:, 3] *= sy

                # Calculate Error Metric: 1.0 - Recall
                num_gt = len(gt_boxes)

                if num_gt == 0:
                    # Negative sample (No findings)
                    # Error is 1.0 if we predicted boxes (False Positive), 0.0 otherwise
                    error = 1.0 if len(boxes_orig) > 0 else 0.0
                    avg_area = 0.0
                    ar = orig_w / orig_h if orig_h > 0 else 0.0
                else:
                    # Positive sample
                    if len(boxes_orig) == 0:
                        recall = 0.0
                    else:
                        # Calculate IoU matrix (N_pred, N_gt)
                        ious = calculate_iou(boxes_orig, gt_boxes)
                        if ious.shape[1] > 0:
                            # For each GT, did we find it? (Max IoU > 0.4)
                            max_ious = np.max(ious, axis=0)
                            hits = np.sum(max_ious > 0.4)
                            recall = hits / num_gt
                        else:
                            recall = 0.0

                    error = 1.0 - recall

                    # Calculate Features
                    areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (
                        gt_boxes[:, 3] - gt_boxes[:, 1]
                    )
                    avg_area = np.mean(areas)
                    ar = orig_w / orig_h if orig_h > 0 else 0.0

                error_magnitudes.append(error)
                feat_area.append(avg_area)
                feat_num_objs.append(num_gt)
                feat_aspect_ratio.append(ar)

    # Compute Correlations
    error_magnitudes = np.array(error_magnitudes)
    feat_area = np.array(feat_area)
    feat_num_objs = np.array(feat_num_objs)
    feat_aspect_ratio = np.array(feat_aspect_ratio)

    # 1. Error vs BBox Area (Only for positive samples)
    valid_area_idx = feat_num_objs > 0
    if np.sum(valid_area_idx) > 10:
        corr_area = np.corrcoef(
            error_magnitudes[valid_area_idx], feat_area[valid_area_idx]
        )[0, 1]
        print(f"Correlation (Error vs BBox Area): {corr_area:.4f}")
    else:
        print("Correlation (Error vs BBox Area): N/A (Insufficient positive samples)")

    # 2. Error vs Num Objects
    corr_num = np.corrcoef(error_magnitudes, feat_num_objs)[0, 1]
    print(f"Correlation (Error vs Num Objects): {corr_num:.4f}")

    # 3. Error vs Aspect Ratio
    corr_ar = np.corrcoef(error_magnitudes, feat_aspect_ratio)[0, 1]
    print(f"Correlation (Error vs Aspect Ratio): {corr_ar:.4f}")


def main():
    # 1. Configuration & Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 2. Data Loading
    # We instantiate datasets manually to enable subsetting for the fast baseline
    train_dataset = ChestXRayDataset(mode="train", transform=get_train_transforms())
    val_dataset = ChestXRayDataset(mode="val", transform=get_val_transforms())
    test_dataset = ChestXRayDataset(mode="test", transform=get_val_transforms())

    # Subsample Training Data
    # Limit to 15,000 samples to ensure execution within 2 hours
    subset_size = min(15000, len(train_dataset))
    print(f"Subsampling training set to {subset_size} images for fast baseline...")
    indices = torch.randperm(len(train_dataset))[:subset_size]
    train_subset = Subset(train_dataset, indices)

    train_loader = DataLoader(
        train_subset,
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
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    # 3. Model Initialization
    model = DetModel(Config).to(device)

    # 4. Optimization
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Reduced epochs for baseline constraint
    EPOCHS = 5

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=Config.MIN_LR
    )

    # 5. Training
    engine = Engine(model, device, optimizer, scheduler)
    engine.run(train_loader, val_loader, test_loader, epochs=EPOCHS)

    # 6. Final Validation
    print("Running Final Validation...")
    final_map = engine.evaluate(val_loader)
    print(f"Final Validation Metric: {final_map}")

    # 7. Failure Analysis
    failure_analysis(engine, val_loader)

    # 8. Submission Generation
    threshold = 0.1783551866
    if final_map > threshold:
        print(f"Validation metric {final_map} > {threshold}. Generating submission...")
        sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        engine.predict_test(test_loader, sub_path)
    else:
        print(
            f"Validation metric {final_map} <= {threshold}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
