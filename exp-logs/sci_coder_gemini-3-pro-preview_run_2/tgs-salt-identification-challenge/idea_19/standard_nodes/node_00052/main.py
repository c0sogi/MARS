import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import warnings

# Import provided library modules
from library.utils import set_seed, calculate_map, calculate_iou, rle_decode, do_unpad
from library.models import ResNet34WideLinkNet
from library.dataset import SaltDataset, get_transforms
from library.engine import SaltEngine
from library.losses import CombinedLoss

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_WORKERS = 4
TEACHER_EPOCHS = 20
STUDENT_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
SUBMISSION_THRESHOLD_REQ = 0.7985
OUTPUT_DIR = "./working"
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")


def main():
    # 1. Setup
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    print(f"Running on device: {DEVICE}")

    # 2. Data Loading (Stage 1)
    print("\n--- Initializing Data Loaders (Stage 1) ---")

    # Teacher Train Set: Weak Augmentation, 50% Depth Masking
    train_ds_teacher = SaltDataset(
        mode="train", transform=get_transforms("train"), depth_mask_prob=0.5
    )
    train_loader_teacher = DataLoader(
        train_ds_teacher,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Validation Set
    val_ds = SaltDataset(mode="val", transform=get_transforms("val"))
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization (Teacher)
    print("\n--- Initializing Teacher Model ---")
    teacher_model = ResNet34WideLinkNet().to(DEVICE)
    optimizer = optim.AdamW(
        teacher_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TEACHER_EPOCHS, eta_min=1e-6
    )
    loss_fn = CombinedLoss(bce_weight=0.5, lovasz_weight=0.5)

    engine = SaltEngine(teacher_model, DEVICE, optimizer, scheduler)

    # 4. Train Teacher
    print(f"\n--- Training Teacher for {TEACHER_EPOCHS} epochs ---")
    best_teacher_map = 0.0
    best_teacher_path = os.path.join(OUTPUT_DIR, "teacher_best.pth")

    for epoch in range(TEACHER_EPOCHS):
        train_loss = engine.train_teacher_epoch(train_loader_teacher, loss_fn)
        val_loss, val_map = engine.validate(val_loader, loss_fn)

        print(
            f"Epoch {epoch+1}/{TEACHER_EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val mAP: {val_map:.4f}"
        )

        if val_map > best_teacher_map:
            best_teacher_map = val_map
            torch.save(teacher_model.state_dict(), best_teacher_path)
            print(f"  -> New Best Teacher Saved! ({best_teacher_map:.4f})")

    # 5. Pseudo-Labeling
    print("\n--- Generating Pseudo-Labels for Noisy Student ---")
    # Load best teacher
    teacher_model.load_state_dict(torch.load(best_teacher_path, map_location=DEVICE))

    # Test Dataset for Pseudo-labeling: Force zero depth (Generalist mode)
    test_ds_pseudo = SaltDataset(
        mode="test", transform=get_transforms("val"), force_zero_depth=True
    )
    test_loader_pseudo = DataLoader(
        test_ds_pseudo, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # Generate soft predictions
    pseudo_labels = engine.predict_proba(test_loader_pseudo)
    print(f"Generated pseudo-labels for {len(pseudo_labels)} test images.")

    # 6. Data Loading (Stage 2 - Student)
    print("\n--- Initializing Data Loaders (Stage 2) ---")

    # Labeled Data for Student: Strong Augmentation, True Depth (Specialist)
    # Note: We use 'student' transform which includes strong augs
    train_ds_student_labeled = SaltDataset(
        mode="train",
        transform=get_transforms("student"),
        depth_mask_prob=0.0,  # Student learns to use depth when available
    )
    loader_student_labeled = DataLoader(
        train_ds_student_labeled,
        batch_size=BATCH_SIZE // 2,  # Split batch between labeled and unlabeled
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # Unlabeled Data for Student: Strong Augmentation, Zero Depth, Soft Targets
    train_ds_student_unlabeled = SaltDataset(
        mode="test",
        transform=get_transforms("student"),
        force_zero_depth=True,
        pseudo_labels=pseudo_labels,
    )
    loader_student_unlabeled = DataLoader(
        train_ds_student_unlabeled,
        batch_size=BATCH_SIZE // 2,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # 7. Model Initialization (Student)
    print("\n--- Initializing Student Model ---")
    student_model = ResNet34WideLinkNet().to(DEVICE)
    optimizer_student = optim.AdamW(
        student_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler_student = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_student, T_max=STUDENT_EPOCHS, eta_min=1e-6
    )

    engine_student = SaltEngine(
        student_model, DEVICE, optimizer_student, scheduler_student
    )

    # 8. Train Student
    print(f"\n--- Training Student for {STUDENT_EPOCHS} epochs ---")
    best_student_map = 0.0
    best_student_path = os.path.join(OUTPUT_DIR, "student_best.pth")

    for epoch in range(STUDENT_EPOCHS):
        # Supervised loss for labeled part
        train_loss = engine_student.train_student_epoch(
            loader_student_labeled, loader_student_unlabeled, loss_fn
        )

        # Validate (Student uses standard validation settings)
        val_loss, val_map = engine_student.validate(val_loader, loss_fn)

        print(
            f"Epoch {epoch+1}/{STUDENT_EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val mAP: {val_map:.4f}"
        )

        if val_map > best_student_map:
            best_student_map = val_map
            torch.save(student_model.state_dict(), best_student_path)
            print(f"  -> New Best Student Saved! ({best_student_map:.4f})")

    # 9. Final Evaluation & Threshold Optimization
    print("\n--- Final Evaluation & Threshold Optimization ---")
    # Load best student
    student_model.load_state_dict(torch.load(best_student_path, map_location=DEVICE))

    # Get raw probabilities for validation set
    val_probs_dict = engine_student.predict_proba(val_loader)

    # Load GT masks for validation
    val_meta = pd.read_csv("./metadata/val.csv")
    val_gts = []
    val_ids = []
    val_preds = []

    for _, row in val_meta.iterrows():
        img_id = row["id"]
        rle = row["rle_mask"]
        if pd.isna(rle) or rle == "":
            mask = np.zeros((101, 101), dtype=np.uint8)
        else:
            mask = rle_decode(rle)
        val_gts.append(mask)
        val_ids.append(img_id)
        val_preds.append(val_probs_dict[img_id])

    # Threshold Sweep
    thresholds = np.arange(0.3, 0.75, 0.05)
    best_thresh = 0.5
    best_sweep_map = 0.0

    for t in thresholds:
        # Binarize
        binary_preds = [(p > t).astype(np.uint8) for p in val_preds]
        score = calculate_map(binary_preds, val_gts)
        if score > best_sweep_map:
            best_sweep_map = score
            best_thresh = t

    print(f"Optimized Threshold: {best_thresh:.2f}")
    print(f"Final Validation Metric: {best_sweep_map:.10f}")

    # 10. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate per-image IoU at best threshold
    ious = []
    for p, g in zip(val_preds, val_gts):
        p_bin = (p > best_thresh).astype(np.uint8)
        ious.append(calculate_iou(p_bin, g))

    val_meta["iou"] = ious
    val_meta["error"] = 1.0 - val_meta["iou"]

    # Correlation with Depth
    corr_depth = val_meta["error"].corr(val_meta["z"])
    # Correlation with Salt Coverage
    corr_coverage = val_meta["error"].corr(val_meta["salt_coverage"])

    print(f"Correlation (Error vs Depth): {corr_depth:.4f}")
    print(f"Correlation (Error vs Salt Coverage): {corr_coverage:.4f}")

    # 11. Submission
    if best_sweep_map > SUBMISSION_THRESHOLD_REQ:
        print(
            f"\nMetric {best_sweep_map:.4f} > {SUBMISSION_THRESHOLD_REQ}. Generating submission..."
        )

        test_ds_final = SaltDataset(
            mode="test",
            transform=get_transforms("val"),
            force_zero_depth=True,  # Inference assumption
        )
        test_loader_final = DataLoader(
            test_ds_final, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
        )

        engine_student.generate_submission(
            test_loader_final, SUBMISSION_PATH, threshold=best_thresh
        )
    else:
        print(
            f"\nMetric {best_sweep_map:.4f} <= {SUBMISSION_THRESHOLD_REQ}. Skipping submission."
        )


if __name__ == "__main__":
    main()
