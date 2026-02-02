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

    # Override Config for efficient execution within time limits
    # 15 epochs is sufficient for convergence with pre-trained ResNet34
    Config.TEACHER_EPOCHS = 15
    Config.STUDENT_EPOCHS = 15

    print(f"Configuration:")
    print(f"  Device: {device}")
    print(f"  Teacher Epochs: {Config.TEACHER_EPOCHS}")
    print(f"  Student Epochs: {Config.STUDENT_EPOCHS}")
    print(f"  Working Dir: {Config.WORKING_DIR}")

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)  # Hold-out validation set
    test_df = pd.read_csv(Config.TEST_CSV)

    # -------------------------------------------------------------------------
    # 1. Stage 1: Train Specialist Teachers (5-Fold Cross Validation)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("STAGE 1: Training Specialist Teachers")
    print("=" * 40)

    # Split training data into 5 folds for ensemble training
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.SEED)
    train_df["fold"] = -1
    for fold, (t_idx, v_idx) in enumerate(
        skf.split(train_df, train_df["coverage_class"])
    ):
        train_df.loc[v_idx, "fold"] = fold

    teacher_checkpoints = []

    for fold in range(5):
        print(f"\n--- Training Teacher Fold {fold} ---")

        # Prepare Fold Data
        fold_train = train_df[train_df["fold"] != fold].reset_index(drop=True)
        fold_val = train_df[train_df["fold"] == fold].reset_index(drop=True)

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
            cache_name=f"train_f{fold}",
        )
        val_ds = SaltDataset(
            fold_val,
            mode="val",
            transform=get_transforms("valid"),
            depth_stats=depth_stats,
            cache_name=f"val_f{fold}",
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
        model = SaltNet(mode="teacher").to(device)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Config.TEACHER_EPOCHS, eta_min=Config.ETA_MIN
        )
        loss_fn = TeacherLoss()

        best_score = 0.0
        best_path = os.path.join(Config.WORKING_DIR, f"teacher_fold{fold}.pth")

        # Training Loop
        for epoch in range(Config.TEACHER_EPOCHS):
            train_loss = Engine.train_teacher_epoch(
                model, train_loader, optimizer, device, loss_fn, scheduler
            )
            val_loss, val_score = Engine.validate(
                model, val_loader, device, loss_fn, mode="teacher"
            )

            if val_score > best_score:
                best_score = val_score
                Engine.save_checkpoint(model, best_path)

        print(f"  Fold {fold} Best mAP: {best_score:.4f}")
        teacher_checkpoints.append(best_path)

        # Cleanup
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 2. Stage 2: Marginalized Pseudo-Labeling
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("STAGE 2: Generating Pseudo-Labels")
    print("=" * 40)

    # Generate soft masks for test images using the teacher ensemble and depth scanning
    pseudo_labels = generate_marginalized_pseudo_labels(
        teacher_checkpoints, device, load_cached_data=True
    )

    # -------------------------------------------------------------------------
    # 3. Stage 3: Train Generalist Student
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("STAGE 3: Training Generalist Student")
    print("=" * 40)

    # Calculate global depth stats for the student (used for normalization)
    global_depth_mean = train_df["z"].mean()
    global_depth_std = train_df["z"].std()
    global_stats = (global_depth_mean, global_depth_std)

    # Labeled Data: Full Train Set
    labeled_ds = SaltDataset(
        train_df,
        mode="train",
        transform=get_transforms("train"),
        depth_stats=global_stats,
        cache_name="full_train",
    )
    labeled_loader = DataLoader(
        labeled_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # Unlabeled Data: Test Set with Pseudo Labels
    unlabeled_ds = PseudoDataset(
        test_df, pseudo_labels, transform=get_transforms("train")
    )
    unlabeled_loader = DataLoader(
        unlabeled_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # Validation Data: Hold-out Val Set
    val_ds = SaltDataset(
        val_df,
        mode="val",
        transform=get_transforms("valid"),
        depth_stats=global_stats,
        cache_name="holdout_val",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Student Model Setup
    student_model = SaltNet(mode="student").to(device)
    optimizer = optim.AdamW(
        student_model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.STUDENT_EPOCHS, eta_min=Config.ETA_MIN
    )
    loss_fn = StudentLoss(depth_weight=0.5)  # Weight for auxiliary depth loss

    best_student_score = 0.0
    student_ckpt_path = os.path.join(Config.WORKING_DIR, "student_best.pth")

    # Student Training Loop
    for epoch in range(Config.STUDENT_EPOCHS):
        train_loss = Engine.train_student_epoch(
            student_model,
            labeled_loader,
            unlabeled_loader,
            optimizer,
            device,
            loss_fn,
            scheduler,
        )
        val_loss, val_score = Engine.validate(
            student_model, val_loader, device, loss_fn, mode="student"
        )

        print(
            f"  Epoch {epoch+1}/{Config.STUDENT_EPOCHS} | Train Loss: {train_loss:.4f} | Val mAP: {val_score:.4f}"
        )

        if val_score > best_student_score:
            best_student_score = val_score
            Engine.save_checkpoint(student_model, student_ckpt_path)

    # Load best student model for final evaluation
    student_model.load_state_dict(torch.load(student_ckpt_path, map_location=device))
    student_model.eval()

    # -------------------------------------------------------------------------
    # 4. Validation & Failure Analysis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("VALIDATION & ANALYSIS")
    print("=" * 40)

    val_depths = {}
    val_scores = {}

    # Threshold optimization variables
    thresholds = np.arange(0.3, 0.75, 0.05)
    thresh_scores = {t: [] for t in thresholds}

    with torch.no_grad():
        for i in range(len(val_ds)):
            img, mask, depth, img_id = val_ds[i]

            # Prepare input
            img_t = img.unsqueeze(0).to(device)  # (1, 1, H, W)

            # Predict
            logits, _ = student_model(img_t)
            prob = torch.sigmoid(logits).cpu().numpy()[0, 0]  # (H, W)

            # Unpad to original size
            prob_orig = unpad_image(prob)
            mask_orig = unpad_image(mask.numpy()[0])

            # Store depth for analysis
            val_depths[img_id] = val_df[val_df["id"] == img_id]["z"].values[0]

            # Calculate mAP at default 0.5 threshold for failure analysis
            score_default = calculate_iou_map(
                (prob_orig > 0.5).astype(np.uint8), mask_orig
            )
            val_scores[img_id] = score_default

            # Collect scores for threshold optimization
            for t in thresholds:
                bin_pred = (prob_orig > t).astype(np.uint8)
                s = calculate_iou_map(bin_pred, mask_orig)
                thresh_scores[t].append(s)

    # Find best threshold
    avg_scores_by_thresh = {t: np.mean(scores) for t, scores in thresh_scores.items()}
    best_threshold = max(avg_scores_by_thresh, key=avg_scores_by_thresh.get)
    final_metric = avg_scores_by_thresh[best_threshold]

    print(f"Optimal Probability Threshold: {best_threshold:.2f}")
    print(f"Final Validation Metric: {final_metric:.10f}")

    # Failure Analysis: Correlation with Depth
    errors = []
    depths_list = []
    for img_id, score in val_scores.items():
        errors.append(1.0 - score)
        depths_list.append(val_depths[img_id])

    correlation = np.corrcoef(errors, depths_list)[0, 1]
    print(f"Correlation (Error vs Depth): {correlation:.4f}")

    # -------------------------------------------------------------------------
    # 5. Submission
    # -------------------------------------------------------------------------
    if final_metric > 0.7985:
        print("\n" + "=" * 40)
        print("GENERATING SUBMISSION")
        print("=" * 40)

        # Setup Test Loader
        test_ds = SaltDataset(
            test_df, mode="test", transform=get_transforms("test"), load_cached=True
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        # Predict with Test-Time Augmentation
        raw_preds = Engine.predict_tta(student_model, test_loader, device)

        submission_rows = []
        for img_id, prob_map in raw_preds.items():
            # Unpad
            prob_orig = unpad_image(prob_map)

            # Binarize with optimal threshold
            mask_bin = (prob_orig > best_threshold).astype(np.uint8)

            # RLE Encode
            rle = rle_encode(mask_bin)
            submission_rows.append({"id": img_id, "rle_mask": rle})

        # Save Submission
        sub_df = pd.DataFrame(submission_rows)
        sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        sub_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")
    else:
        print(
            f"Validation metric {final_metric:.4f} is below threshold 0.7985. Skipping submission."
        )


if __name__ == "__main__":
    main()
