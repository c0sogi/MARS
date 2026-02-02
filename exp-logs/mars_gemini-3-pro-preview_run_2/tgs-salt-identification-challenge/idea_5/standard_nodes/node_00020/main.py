import os
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.utils import set_seed, rle_encode, calc_iou_metric
from library.dataset import load_data, SaltDataset, get_transforms, get_depth_stats
from library.model import DepthAwareLinkNet
from library.losses import BCEDiceLoss
from library.train_eval import train_one_epoch, validate, predict_with_tta


def crop_center(img_batch, target_h=101, target_w=101):
    """
    Crops the center of the image batch.
    Assumes input shape (N, C, H, W).
    """
    _, _, h, w = img_batch.shape
    start_h = (h - target_h) // 2
    start_w = (w - target_w) // 2
    return img_batch[:, :, start_h : start_h + target_h, start_w : start_w + target_w]


def main():
    # 1. Setup
    start_time = time.time()
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")

    # Override Config for Fast Baseline
    # 2400 images is small, but to ensure we finish comfortably in 2 hours with 3 phases:
    Config.EPOCHS = 15
    Config.RETRAIN_EPOCHS = 15
    Config.N_FOLDS = 5

    # Ensure directories exist
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"Running on {device}")
    print(f"Config: {Config.N_FOLDS} Folds, {Config.EPOCHS} Epochs/Phase")

    # 2. Load Data
    print("Loading Data...")
    # Load cached numpy arrays
    train_data = load_data(Config.TRAIN_CSV, "train", load_cached_data=True)
    val_data = load_data(Config.VAL_CSV, "val", load_cached_data=True)
    test_data = load_data(Config.TEST_CSV, "test", load_cached_data=True)

    # Depth Stats for normalization
    depth_mean, depth_std = get_depth_stats()
    depth_stats = (depth_mean, depth_std)

    # 3. Phase 1: Supervised Training (5-Fold CV)
    print("\n=== Phase 1: Supervised Training (5-Fold CV) ===")

    kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)
    phase1_models = []

    # Training on train_data
    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_data["images"])):
        print(f"Fold {fold + 1}/{Config.N_FOLDS}")

        # Create Datasets
        train_sub_data = {
            "images": train_data["images"][train_idx],
            "masks": train_data["masks"][train_idx],
            "depths": train_data["depths"][train_idx],
            "ids": train_data["ids"][train_idx],
        }
        valid_sub_data = {
            "images": train_data["images"][valid_idx],
            "masks": train_data["masks"][valid_idx],
            "depths": train_data["depths"][valid_idx],
            "ids": train_data["ids"][valid_idx],
        }

        train_dataset = SaltDataset(
            train_sub_data, transform=get_transforms("train"), depth_stats=depth_stats
        )
        valid_dataset = SaltDataset(
            valid_sub_data, transform=get_transforms("val"), depth_stats=depth_stats
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        model = DepthAwareLinkNet().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        criterion = BCEDiceLoss()

        best_map = -1.0
        best_model_path = os.path.join(Config.CHECKPOINT_DIR, f"phase1_fold_{fold}.pth")

        for epoch in range(Config.EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, val_map, _ = validate(model, valid_loader, criterion, device)

            if val_map > best_map:
                best_map = val_map
                torch.save(model.state_dict(), best_model_path)

        # Load best model
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        phase1_models.append(model)

        # Clean up
        del (
            train_dataset,
            valid_dataset,
            train_loader,
            valid_loader,
            optimizer,
            criterion,
            model,
        )
        torch.cuda.empty_cache()

    # 4. Phase 2: Pseudo-Labeling
    print("\n=== Phase 2: Pseudo-Labeling ===")

    test_dataset = SaltDataset(
        test_data, transform=get_transforms("test"), depth_stats=depth_stats
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    test_preds_accum = np.zeros(
        (len(test_data["images"]), 1, Config.IMG_SIZE, Config.IMG_SIZE),
        dtype=np.float32,
    )

    for model in phase1_models:
        model = model.to(device)
        preds, _ = predict_with_tta(model, test_loader, device)
        test_preds_accum += preds
        model.to("cpu")  # Move back to CPU to save GPU memory

    test_preds_avg = test_preds_accum / len(phase1_models)
    test_preds_cropped = crop_center(test_preds_avg)

    # Generate Pseudo-Labels (Hard Threshold 0.5)
    test_masks_pseudo = (test_preds_cropped > 0.5).astype(np.uint8)

    # 5. Phase 3: Retraining
    print("\n=== Phase 3: Retraining with Pseudo-Labels ===")

    phase3_models = []

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_data["images"])):
        print(f"Retraining Fold {fold + 1}/{Config.N_FOLDS}")

        # Original Train Data
        t_imgs = train_data["images"][train_idx]
        t_masks = train_data["masks"][train_idx]
        t_depths = train_data["depths"][train_idx]
        t_ids = train_data["ids"][train_idx]

        # Pseudo Data
        p_masks = test_masks_pseudo.squeeze(1)

        combined_images = np.concatenate([t_imgs, test_data["images"]], axis=0)
        combined_masks = np.concatenate([t_masks, p_masks], axis=0)
        combined_depths = np.concatenate([t_depths, test_data["depths"]], axis=0)
        combined_ids = np.concatenate([t_ids, test_data["ids"]], axis=0)

        # Validation Data (Original internal val)
        v_imgs = train_data["images"][valid_idx]
        v_masks = train_data["masks"][valid_idx]
        v_depths = train_data["depths"][valid_idx]
        v_ids = train_data["ids"][valid_idx]

        train_dataset = SaltDataset(
            {
                "images": combined_images,
                "masks": combined_masks,
                "depths": combined_depths,
                "ids": combined_ids,
            },
            transform=get_transforms("train"),
            depth_stats=depth_stats,
        )

        valid_dataset = SaltDataset(
            {"images": v_imgs, "masks": v_masks, "depths": v_depths, "ids": v_ids},
            transform=get_transforms("val"),
            depth_stats=depth_stats,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        model = DepthAwareLinkNet().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        criterion = BCEDiceLoss()

        best_map = -1.0
        best_model_path = os.path.join(Config.CHECKPOINT_DIR, f"phase3_fold_{fold}.pth")

        for epoch in range(Config.RETRAIN_EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, val_map, _ = validate(model, valid_loader, criterion, device)

            if val_map > best_map:
                best_map = val_map
                torch.save(model.state_dict(), best_model_path)

        model.load_state_dict(torch.load(best_model_path, map_location=device))
        phase3_models.append(model)

        del (
            train_dataset,
            valid_dataset,
            train_loader,
            valid_loader,
            optimizer,
            criterion,
            model,
        )
        torch.cuda.empty_cache()

    # 6. Final Validation on Hold-Out Set
    print("\n=== Final Validation ===")

    val_dataset_holdout = SaltDataset(
        val_data, transform=get_transforms("val"), depth_stats=depth_stats
    )
    val_loader_holdout = DataLoader(
        val_dataset_holdout,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    val_preds_accum = np.zeros(
        (len(val_data["images"]), 1, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32
    )

    for model in phase3_models:
        model = model.to(device)
        preds, _ = predict_with_tta(model, val_loader_holdout, device)
        val_preds_accum += preds
        model.to("cpu")

    val_preds_avg = val_preds_accum / len(phase3_models)
    val_preds_cropped = crop_center(val_preds_avg)

    # Find best threshold on validation set
    thresholds = np.arange(0.3, 0.76, 0.05)
    best_final_score = -1.0
    best_final_threshold = 0.5
    y_true = val_data["masks"]

    for t in thresholds:
        score = calc_iou_metric(val_preds_cropped, y_true, binarization_threshold=t)
        if score > best_final_score:
            best_final_score = score
            best_final_threshold = t

    print(f"Final Validation Metric: {best_final_score}")

    # 7. Failure Analysis
    print("\n=== Failure Analysis ===")
    y_pred_bin = (val_preds_cropped.squeeze(1) > best_final_threshold).astype(np.uint8)

    ious = []
    for i in range(len(y_true)):
        t_mask = y_true[i]
        p_mask = y_pred_bin[i]

        intersection = np.logical_and(t_mask, p_mask).sum()
        union = np.logical_or(t_mask, p_mask).sum()

        if union == 0:
            iou = 1.0
        else:
            iou = intersection / union
        ious.append(iou)

    ious = np.array(ious)
    errors = 1.0 - ious

    corr_depth, _ = pearsonr(errors, val_data["depths"])
    print(f"Correlation (Error vs Depth): {corr_depth:.4f}")

    coverages = np.array([np.sum(m) / (101 * 101) for m in y_true])
    corr_cov, _ = pearsonr(errors, coverages)
    print(f"Correlation (Error vs Salt Coverage): {corr_cov:.4f}")

    # 8. Submission
    SUBMISSION_THRESHOLD = 0.7916666666666666
    if best_final_score > SUBMISSION_THRESHOLD:
        print("\n=== Generating Submission ===")

        test_preds_accum = np.zeros(
            (len(test_data["images"]), 1, Config.IMG_SIZE, Config.IMG_SIZE),
            dtype=np.float32,
        )

        for model in phase3_models:
            model = model.to(device)
            preds, _ = predict_with_tta(model, test_loader, device)
            test_preds_accum += preds
            model.to("cpu")

        test_preds_avg = test_preds_accum / len(phase3_models)
        test_preds_cropped = crop_center(test_preds_avg)

        test_masks_binary = (test_preds_cropped > best_final_threshold).astype(np.uint8)
        test_masks_binary = test_masks_binary.squeeze(1)

        rles = []
        ids = test_data["ids"]
        for i in range(len(ids)):
            rle = rle_encode(test_masks_binary[i])
            rles.append(rle)

        sub_df = pd.DataFrame({"id": ids, "rle_mask": rles})
        sub_df.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")
    else:
        print(
            f"Validation score {best_final_score} did not meet threshold {SUBMISSION_THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
