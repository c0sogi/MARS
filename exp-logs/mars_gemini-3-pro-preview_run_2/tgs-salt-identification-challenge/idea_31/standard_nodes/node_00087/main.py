import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

# Import library modules
from library.config import Config
from library.utils import rle_encode, calculate_iou_map, unpad_image, pad_image
from library.dataset import SaltDataset, get_transforms
from library.models import SaltNet
from library.losses import TeacherLoss, StudentLoss
from library.engine import Engine
from library.distillation import generate_marginalized_pseudo_labels, PseudoDataset


def main():
    # -------------------------------------------------------------------------
    # 0. Setup & Configuration
    # -------------------------------------------------------------------------
    Engine.set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Use 50 epochs for robust convergence (Cite solution_lesson_node_00035)
    Config.EPOCHS = 50

    print(f"Configuration:")
    print(f"  Device: {device}")
    print(f"  Epochs: {Config.EPOCHS}")
    print(f"  Working Dir: {Config.WORKING_DIR}")

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Load Metadata
    # Merge train and val CSVs to do a proper full 5-Fold CV
    train_meta = pd.read_csv(Config.TRAIN_CSV)
    val_meta = pd.read_csv(Config.VAL_CSV)
    full_train_df = pd.concat([train_meta, val_meta], ignore_index=True)

    test_df = pd.read_csv(Config.TEST_CSV)

    # -------------------------------------------------------------------------
    # 1. 5-Fold Cross-Validation Training
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("5-FOLD CROSS-VALIDATION TRAINING")
    print("=" * 40)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.SEED)
    full_train_df["fold"] = -1
    for fold, (t_idx, v_idx) in enumerate(
        skf.split(full_train_df, full_train_df["coverage_class"])
    ):
        full_train_df.loc[v_idx, "fold"] = fold

    fold_models = []
    oof_preds = {}  # id -> prob_map
    oof_targets = {}  # id -> mask
    oof_depths = {}  # id -> depth

    for fold in range(5):
        print(f"\n--- Training Fold {fold} ---")

        # Prepare Fold Data
        fold_train = full_train_df[full_train_df["fold"] != fold].reset_index(drop=True)
        fold_val = full_train_df[full_train_df["fold"] == fold].reset_index(drop=True)

        # Calculate depth stats for this fold's training data
        depth_mean = fold_train["z"].mean()
        depth_std = fold_train["z"].std()
        depth_stats = (depth_mean, depth_std)

        # Create Datasets & Loaders
        train_ds = SaltDataset(
            fold_train,
            mode="train",
            transform=get_transforms("train"),
            depth_stats=depth_stats,
            cache_name=f"train_f{fold}_full",
        )
        val_ds = SaltDataset(
            fold_val,
            mode="val",
            transform=get_transforms("valid"),
            depth_stats=depth_stats,
            cache_name=f"val_f{fold}_full",
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Model & Optimization
        model = SaltNet().to(device)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
        )
        loss_fn = TeacherLoss()

        best_score = 0.0
        best_path = os.path.join(Config.WORKING_DIR, f"model_fold{fold}.pth")

        # Training Loop
        for epoch in range(Config.EPOCHS):
            train_loss = Engine.train_epoch(
                model, train_loader, optimizer, device, loss_fn, scheduler
            )
            val_loss, val_score = Engine.validate(model, val_loader, device, loss_fn)

            if val_score > best_score:
                best_score = val_score
                Engine.save_checkpoint(model, best_path)

            # Optional: Print progress every 5 epochs
            if (epoch + 1) % 10 == 0:
                print(
                    f"  Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val mAP: {val_score:.4f}"
                )

        print(f"  Fold {fold} Best mAP: {best_score:.4f}")
        fold_models.append(best_path)

        # Generate OOF Predictions for this fold using best model
        model.load_state_dict(torch.load(best_path, map_location=device))
        model.eval()

        with torch.no_grad():
            for i in range(len(val_ds)):
                img, mask, depth, img_id = val_ds[i]

                # Inference
                img_t = img.unsqueeze(0).to(device)
                depth_t = depth.unsqueeze(0).to(device)

                out = model(img_t, depth_t)
                prob = torch.sigmoid(out).cpu().numpy()[0, 0]

                # Unpad
                prob_orig = unpad_image(prob)
                mask_orig = unpad_image(mask.numpy()[0])

                oof_preds[img_id] = prob_orig
                oof_targets[img_id] = mask_orig
                oof_depths[img_id] = full_train_df[full_train_df["id"] == img_id][
                    "z"
                ].values[0]

        # Cleanup
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 2. Global Validation & Failure Analysis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("GLOBAL OOF VALIDATION & ANALYSIS")
    print("=" * 40)

    # Convert to arrays
    all_preds = []
    all_targets = []
    all_ids = sorted(list(oof_preds.keys()))

    for img_id in all_ids:
        all_preds.append(oof_preds[img_id])
        all_targets.append(oof_targets[img_id])

    # Threshold Optimization on Global OOF
    thresholds = np.arange(0.3, 0.75, 0.05)
    best_global_score = 0.0
    best_global_thresh = 0.5

    for t in thresholds:
        scores = []
        for i in range(len(all_preds)):
            bin_pred = (all_preds[i] > t).astype(np.uint8)
            s = calculate_iou_map(bin_pred, all_targets[i])
            scores.append(s)
        mean_score = np.mean(scores)
        if mean_score > best_global_score:
            best_global_score = mean_score
            best_global_thresh = t

    print(f"Optimal Probability Threshold: {best_global_thresh:.2f}")
    print(f"Final Global OOF Metric: {best_global_score:.10f}")

    # Failure Analysis
    errors = []
    depths_list = []
    for img_id in all_ids:
        # Calculate score at best threshold
        bin_pred = (oof_preds[img_id] > best_global_thresh).astype(np.uint8)
        s = calculate_iou_map(bin_pred, oof_targets[img_id])
        errors.append(1.0 - s)
        depths_list.append(oof_depths[img_id])

    correlation = np.corrcoef(errors, depths_list)[0, 1]
    print(f"Correlation (Error vs Depth): {correlation:.4f}")

    # -------------------------------------------------------------------------
    # 3. Submission (Ensemble)
    # -------------------------------------------------------------------------
    if best_global_score > 0.7985:
        print("\n" + "=" * 40)
        print("GENERATING ENSEMBLE SUBMISSION")
        print("=" * 40)

        # We need to process test set with depth stats from full training set
        global_depth_mean = full_train_df["z"].mean()
        global_depth_std = full_train_df["z"].std()

        test_ds = SaltDataset(
            test_df,
            mode="test",
            transform=get_transforms("test"),
            load_cached=True,
            depth_stats=(global_depth_mean, global_depth_std),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        # Ensemble Accumulator
        ensemble_preds = {}  # id -> sum_prob
        for img_id in test_df["id"].values:
            ensemble_preds[img_id] = 0.0

        for fold, model_path in enumerate(fold_models):
            print(f"Inference with model fold {fold}...")
            model = SaltNet().to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))

            # Predict with TTA
            fold_preds = Engine.predict_tta(model, test_loader, device)

            for img_id, prob in fold_preds.items():
                ensemble_preds[img_id] += prob

            del model
            torch.cuda.empty_cache()

        # Average and Save
        submission_rows = []
        for img_id, sum_prob in ensemble_preds.items():
            avg_prob = sum_prob / 5.0

            # Unpad
            prob_orig = unpad_image(avg_prob)

            # Binarize
            mask_bin = (prob_orig > best_global_thresh).astype(np.uint8)

            # RLE
            rle = rle_encode(mask_bin)
            submission_rows.append({"id": img_id, "rle_mask": rle})

        sub_df = pd.DataFrame(submission_rows)
        sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        sub_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")
    else:
        print(
            f"Validation metric {best_global_score:.4f} is below threshold 0.7985. Skipping submission."
        )


if __name__ == "__main__":
    main()
