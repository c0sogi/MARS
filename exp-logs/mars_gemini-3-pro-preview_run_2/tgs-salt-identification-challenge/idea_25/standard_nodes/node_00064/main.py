import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import set_seed, rle_encode, calculate_iou_batch
from library.dataset import SaltDataset, PseudoLabelDataset
from library.model import ResNet34WideLinkNetMTL
from library.losses import CombinedMTLLoss
from library.engine import (
    train_teacher_epoch,
    train_student_epoch,
    validate,
    predict,
)


def main():
    # 1. Setup
    print("--- Setting up Environment ---")
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Override Config epochs for fast baseline execution (2 hour limit)
    # 30 epochs per stage is sufficient for convergence with OneCycleLR
    EPOCHS_STAGE1 = 30
    EPOCHS_STAGE3 = 30

    # 2. Data Loading
    print("--- Loading Labeled Data ---")
    train_dataset = SaltDataset(mode="train", load_cached_data=True)
    val_dataset = SaltDataset(mode="val", load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
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
    )

    # 3. Stage 1: Train Teacher
    print(f"\n=== Stage 1: Training Teacher Model ({EPOCHS_STAGE1} Epochs) ===")
    teacher_model = ResNet34WideLinkNetMTL().to(device)
    criterion = CombinedMTLLoss().to(device)
    optimizer = optim.AdamW(
        teacher_model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    # Scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS_STAGE1,
        pct_start=0.3,
    )

    best_teacher_map = 0.0
    teacher_save_path = os.path.join(Config.CHECKPOINT_DIR, "best_teacher.pth")

    for epoch in range(1, EPOCHS_STAGE1 + 1):
        train_metrics = train_teacher_epoch(
            teacher_model, train_loader, optimizer, criterion, device, epoch
        )
        scheduler.step()

        # Validate (Force zero depth to simulate test condition capability)
        val_metrics = validate(
            teacher_model, val_loader, criterion, device, force_zero_depth=True
        )

        if val_metrics["mAP"] > best_teacher_map:
            best_teacher_map = val_metrics["mAP"]
            torch.save(teacher_model.state_dict(), teacher_save_path)
            print(f"  -> New Best Teacher mAP: {best_teacher_map:.4f}")

    print(f"Stage 1 Complete. Best Teacher mAP: {best_teacher_map:.4f}")

    # 4. Stage 2: Pseudo-Label Generation
    print("\n=== Stage 2: Generating Soft Pseudo-Labels ===")
    # Load best teacher
    teacher_model.load_state_dict(torch.load(teacher_save_path, map_location=device))

    # Initialize Test Dataset (No soft labels yet)
    test_dataset_inference = PseudoLabelDataset(soft_labels=None, load_cached_data=True)
    test_loader_inference = DataLoader(
        test_dataset_inference,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Generate predictions (returns dict: id -> probability map)
    soft_pseudo_labels = predict(teacher_model, test_loader_inference, device)
    print(f"Generated {len(soft_pseudo_labels)} pseudo-labels.")

    # 5. Stage 3: Train Student (Noisy Student)
    print(f"\n=== Stage 3: Training Student Model ({EPOCHS_STAGE3} Epochs) ===")

    # Initialize Student
    student_model = ResNet34WideLinkNetMTL().to(device)
    optimizer_s = optim.AdamW(
        student_model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    # Dataset with Soft Labels
    unlabeled_dataset = PseudoLabelDataset(
        soft_labels=soft_pseudo_labels, load_cached_data=True
    )
    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,  # Shuffle for training
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    scheduler_s = optim.lr_scheduler.OneCycleLR(
        optimizer_s,
        max_lr=Config.LEARNING_RATE,
        steps_per_epoch=len(train_loader),  # Based on labeled loader length
        epochs=EPOCHS_STAGE3,
        pct_start=0.3,
    )

    best_student_map = 0.0
    student_save_path = os.path.join(Config.CHECKPOINT_DIR, "best_student.pth")

    for epoch in range(1, EPOCHS_STAGE3 + 1):
        train_metrics = train_student_epoch(
            student_model,
            train_loader,
            unlabeled_loader,
            optimizer_s,
            criterion,
            device,
            epoch,
        )
        scheduler_s.step()

        # Validate Student (Force zero depth)
        val_metrics = validate(
            student_model, val_loader, criterion, device, force_zero_depth=True
        )

        if val_metrics["mAP"] > best_student_map:
            best_student_map = val_metrics["mAP"]
            torch.save(student_model.state_dict(), student_save_path)
            print(f"  -> New Best Student mAP: {best_student_map:.4f}")

    # 6. Threshold Optimization & Final Validation
    print("\n=== Threshold Optimization & Final Evaluation ===")
    # Load best student
    student_model.load_state_dict(torch.load(student_save_path, map_location=device))
    student_model.eval()

    # Collect all validation predictions and targets
    all_preds = []
    all_targets = []
    all_depths = []
    all_ids = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].cpu().numpy()
            depths = batch["depth"].cpu().numpy()
            ids = batch["id"]

            # Predict with zero depth (Generalist mode)
            input_depths = torch.zeros_like(batch["depth"]).to(device)
            outputs = student_model(images, input_depths)
            probs = torch.sigmoid(outputs["mask"]).cpu().numpy()

            all_preds.append(probs)
            all_targets.append(masks)
            all_depths.append(depths)
            all_ids.extend(ids)

    all_preds = np.concatenate(all_preds, axis=0)  # (N, 1, H, W)
    all_targets = np.concatenate(all_targets, axis=0)  # (N, 1, H, W)
    all_depths = np.concatenate(all_depths, axis=0).flatten()  # (N,)

    # Search for best threshold
    thresholds = np.arange(0.2, 0.8, 0.05)
    best_thresh = 0.5
    best_final_map = 0.0

    # Helper to calc mAP for a specific binary threshold
    def get_map_score(preds_bin, targets):
        # preds_bin: (N, 1, H, W) binary
        iou_thresholds = Config.IOU_THRESHOLDS
        map_sum = 0.0
        n = preds_bin.shape[0]

        # Flatten for IoU calc
        p_flat = preds_bin.reshape(n, -1)
        t_flat = targets.reshape(n, -1)

        intersection = (p_flat * t_flat).sum(axis=1)
        union = p_flat.sum(axis=1) + t_flat.sum(axis=1) - intersection

        # Avoid div by zero
        ious = np.ones(n)
        non_empty = union > 0
        ious[non_empty] = intersection[non_empty] / union[non_empty]

        # Calculate mAP over IoU thresholds (0.5 to 0.95)
        # For each image, average precision is mean of (iou > t)
        # Vectorized: compare (N, 1) ious with (1, T) thresholds
        ious_exp = ious[:, None]
        thresh_exp = iou_thresholds[None, :]
        hits = (ious_exp > thresh_exp).astype(float)
        ap_per_image = hits.mean(axis=1)
        return ap_per_image.mean()

    print("Optimizing Threshold...")
    for t in thresholds:
        bin_preds = (all_preds > t).astype(np.uint8)
        score = get_map_score(bin_preds, all_targets)
        if score > best_final_map:
            best_final_map = score
            best_thresh = t

    print(f"Best Threshold: {best_thresh:.2f}")
    print(f"Final Validation Metric: {best_final_map}")

    # 7. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate per-image IoU/Error at best threshold
    bin_preds_opt = (all_preds > best_thresh).astype(np.uint8)

    # Calculate IoU per image (scalar)
    p_flat = bin_preds_opt.reshape(len(all_preds), -1)
    t_flat = all_targets.reshape(len(all_targets), -1)
    intersection = (p_flat * t_flat).sum(axis=1)
    union = p_flat.sum(axis=1) + t_flat.sum(axis=1) - intersection
    ious = np.ones(len(all_preds))
    non_empty = union > 0
    ious[non_empty] = intersection[non_empty] / union[non_empty]

    errors = 1.0 - ious

    # Calculate Salt Coverage per image (from target)
    salt_coverage = t_flat.mean(axis=1)

    # Correlations
    df_fail = pd.DataFrame(
        {"error": errors, "depth": all_depths, "coverage": salt_coverage}
    )

    corr_depth = df_fail["error"].corr(df_fail["depth"])
    corr_cov = df_fail["error"].corr(df_fail["coverage"])

    print(f"Correlation (Error vs Depth): {corr_depth:.4f}")
    print(f"Correlation (Error vs Salt Coverage): {corr_cov:.4f}")

    # 8. Submission
    if best_final_map > 0.7985:
        print("\n=== Generating Submission ===")
        # Predict on Test Set using Best Student and Best Threshold
        # Use TTA prediction function which returns probabilities
        test_preds_dict = predict(student_model, test_loader_inference, device)

        submission_data = []

        # Original size for cropping/resizing handled in predict?
        # Predict returns cropped 101x101 probability maps.

        for img_id, prob_map in test_preds_dict.items():
            # Binarize
            mask = (prob_map > best_thresh).astype(np.uint8)

            # Encode
            rle = rle_encode(mask)
            submission_data.append({"id": img_id, "rle_mask": rle})

        submission_df = pd.DataFrame(submission_data)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation Metric {best_final_map:.4f} did not meet threshold 0.7985. Skipping submission."
        )


if __name__ == "__main__":
    main()
