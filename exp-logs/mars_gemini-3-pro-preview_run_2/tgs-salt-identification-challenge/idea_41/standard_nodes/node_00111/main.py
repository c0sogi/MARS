import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.data import get_train_val_loaders, get_test_loader, get_pseudo_loader
from library.models import SpecialistTeacher, GeneralistStudent
from library.training import (
    train_model,
    predict_teacher_marginalized,
    generate_submission,
    validate_epoch,
)
from library.utils import unpad_image, calc_map_score


def main():
    # 1. Setup and Config Overrides for Fast Baseline
    print("Initializing Fast Baseline Run...")
    Config.setup()

    # Override Config for speed (Time limit ~22 mins)
    Config.MAX_SAMPLES = 2000  # Limit dataset size
    Config.EPOCHS_TEACHER = 5  # Reduce epochs
    Config.EPOCHS_STUDENT = 5  # Reduce epochs
    Config.FOLDS = 1  # Train only 1 fold for baseline

    device = Config.DEVICE
    print(f"Device: {device}")
    print(f"Max Samples: {Config.MAX_SAMPLES}")
    print(f"Epochs (Teacher/Student): {Config.EPOCHS_TEACHER}/{Config.EPOCHS_STUDENT}")

    # 2. Data Loading
    print("\nLoading Data...")
    # load_cached_data is handled inside get_train_val_loaders via process_and_cache_data
    train_loader, val_loader, depth_stats = get_train_val_loaders()
    test_loader = get_test_loader()

    # 3. Stage 1: Train Specialist Teacher
    print("\n=== Stage 1: Training Specialist Teacher ===")
    teacher_model = SpecialistTeacher().to(device)

    optimizer_t = optim.AdamW(
        teacher_model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )
    scheduler_t = CosineAnnealingLR(
        optimizer_t, T_max=Config.EPOCHS_TEACHER, eta_min=1e-6
    )

    teacher_save_path = os.path.join(Config.CHECKPOINT_DIR, "teacher_baseline.pth")

    teacher_model = train_model(
        model=teacher_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer_t,
        scheduler=scheduler_t,
        device=device,
        epochs=Config.EPOCHS_TEACHER,
        save_path=teacher_save_path,
        is_teacher=True,
        patience=5,
    )

    # 4. Stage 2: Marginalized Pseudo-Labeling
    print("\n=== Stage 2: Generating Marginalized Soft Pseudo-Labels ===")
    # We use the trained teacher to generate soft targets
    # Note: In a full run we would ensemble, here we use the single baseline teacher
    soft_masks_dict = predict_teacher_marginalized(
        teacher_model, test_loader, device, Config.DEPTH_SCAN_SIGMAS
    )

    # Free up memory
    del teacher_model, optimizer_t, scheduler_t
    torch.cuda.empty_cache()

    # 5. Stage 3: Train Generalist Student
    print("\n=== Stage 3: Training Generalist Student ===")
    pseudo_loader = get_pseudo_loader(soft_masks_dict)

    student_model = GeneralistStudent().to(device)

    optimizer_s = optim.AdamW(
        student_model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )
    scheduler_s = CosineAnnealingLR(
        optimizer_s, T_max=Config.EPOCHS_STUDENT, eta_min=1e-6
    )

    student_save_path = os.path.join(Config.CHECKPOINT_DIR, "student_best.pth")

    student_model = train_model(
        model=student_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer_s,
        scheduler=scheduler_s,
        device=device,
        epochs=Config.EPOCHS_STUDENT,
        save_path=student_save_path,
        is_teacher=False,
        unlabeled_loader=pseudo_loader,
        patience=5,
    )

    # 6. Final Validation
    print("\n=== Final Validation ===")
    # Calculate metric on full validation set
    final_score = validate_epoch(student_model, val_loader, device, is_teacher=False)
    print(f"Final Validation Metric: {final_score:.16f}")

    # 7. Failure Analysis
    print("\n=== Failure Analysis ===")
    student_model.eval()

    val_ious = []
    val_depths = []
    val_coverages = []

    # We need to manually iterate to get per-sample metrics and metadata
    # The val_loader returns (images, masks, depths)
    # We also need salt coverage, which isn't directly in the loader batch,
    # but we can infer it from the mask.

    with torch.no_grad():
        for batch in val_loader:
            images, masks, depths = batch
            images = images.to(device)

            # Student prediction
            logits = student_model(images)
            preds = torch.sigmoid(logits)

            preds_np = preds.cpu().numpy()
            masks_np = masks.numpy()
            depths_np = depths.numpy()  # These are normalized depths

            for p, t, d in zip(preds_np, masks_np, depths_np):
                # Unpad
                p_orig = unpad_image(p[0], Config.ORIG_SIZE)
                t_orig = unpad_image(t[0], Config.ORIG_SIZE)

                # Calculate AP for this single image (mean over thresholds)
                # We reuse calc_map_score but pass single items as batch of 1
                score = calc_map_score(p_orig[None, ...], t_orig[None, ...])
                val_ious.append(score)

                # Metadata
                val_depths.append(d)
                val_coverages.append(np.mean(t_orig))

    val_ious = np.array(val_ious)
    val_depths = np.array(val_depths)
    val_coverages = np.array(val_coverages)

    # Error magnitude = 1 - mAP
    errors = 1.0 - val_ious

    # Correlations
    # Handle case where std is 0 (e.g. all errors same)
    if np.std(errors) > 1e-9:
        corr_depth, _ = pearsonr(val_depths, errors)
        corr_cov, _ = pearsonr(val_coverages, errors)
    else:
        corr_depth, corr_cov = 0.0, 0.0

    print(f"Correlation (Error vs Depth): {corr_depth:.4f}")
    print(f"Correlation (Error vs Salt Coverage): {corr_cov:.4f}")

    if corr_depth > 0.1:
        print("Analysis: Model struggles more with deeper images.")
    elif corr_depth < -0.1:
        print("Analysis: Model struggles more with shallower images.")

    if corr_cov > 0.1:
        print("Analysis: Model struggles more with larger salt structures.")
    elif corr_cov < -0.1:
        print("Analysis: Model struggles more with small/empty salt structures.")

    # 8. Submission Logic
    submission_threshold = 0.7985
    if final_score > submission_threshold:
        print(
            f"\nValidation score ({final_score:.4f}) exceeds threshold ({submission_threshold}). Generating submission..."
        )
        submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        generate_submission(
            model=student_model,
            test_loader=test_loader,
            val_loader=val_loader,
            device=device,
            output_path=submission_path,
        )
    else:
        print(
            f"\nValidation score ({final_score:.4f}) did not exceed threshold ({submission_threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
