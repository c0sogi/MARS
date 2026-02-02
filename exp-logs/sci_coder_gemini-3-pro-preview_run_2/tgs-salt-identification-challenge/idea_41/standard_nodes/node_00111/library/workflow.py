import os
import torch
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.data import get_train_val_loaders, get_test_loader, get_pseudo_loader
from library.models import SpecialistTeacher, GeneralistStudent
from library.training import (
    train_model,
    predict_teacher_marginalized,
    generate_submission,
)


def run_stage1_teacher():
    """
    Stage 1: Train the Specialist Teacher Ensemble.
    Trains multiple instances of the Depth-Injected Teacher model.

    Returns:
        list: Paths to the saved checkpoints of valid teacher models (mAP > 0.75).
    """
    print("=== Stage 1: Training Specialist Teacher Ensemble ===")
    Config.setup()
    device = Config.DEVICE

    # Load Data
    train_loader, val_loader, _ = get_train_val_loaders()

    valid_model_paths = []

    for fold in range(Config.FOLDS):
        print(f"\n--- Training Teacher Fold {fold + 1}/{Config.FOLDS} ---")

        # Initialize Model
        model = SpecialistTeacher().to(device)

        # Optimizer & Scheduler
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = CosineAnnealingLR(
            optimizer, T_max=Config.EPOCHS_TEACHER, eta_min=1e-6
        )

        # Checkpoint Path
        save_path = os.path.join(Config.CHECKPOINT_DIR, f"teacher_fold_{fold}.pth")

        # Train
        model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epochs=Config.EPOCHS_TEACHER,
            save_path=save_path,
            is_teacher=True,
            unlabeled_loader=None,
        )

        # Validation Check (Gating)
        # We reload the best model to check its score, or rely on the fact train_model returns best state
        # Ideally we check the score returned by validation.
        # train_model prints metrics, but here we need to decide whether to keep it.
        # We'll run a quick validation pass to get the exact score for gating logic.
        from library.training import validate_epoch

        final_score = validate_epoch(model, val_loader, device, is_teacher=True)
        print(f"Fold {fold} Final Validation mAP: {final_score:.16f}")

        if final_score >= 0.75:
            valid_model_paths.append(save_path)
            print(f"Fold {fold} accepted.")
        else:
            print(f"Fold {fold} discarded (mAP < 0.75).")

        # Cleanup
        del model, optimizer, scheduler
        torch.cuda.empty_cache()

    return valid_model_paths


def run_stage2_marginalization(teacher_paths):
    """
    Stage 2: Marginalized Soft Pseudo-Labeling.
    Scans valid teacher models across a spectrum of depths to generate robust soft targets.

    Args:
        teacher_paths (list): List of paths to trained teacher checkpoints.

    Returns:
        dict: Dictionary mapping image IDs to averaged soft probability masks.
    """
    print("\n=== Stage 2: Generating Marginalized Soft Pseudo-Labels ===")
    device = Config.DEVICE
    test_loader = get_test_loader()

    # Dictionary to accumulate predictions: {id: accumulated_mask}
    accumulated_preds = {}
    counts = 0

    if not teacher_paths:
        raise RuntimeError("No valid teacher models available for Stage 2.")

    for path in teacher_paths:
        print(f"Processing with Teacher: {os.path.basename(path)}")

        # Load Model
        model = SpecialistTeacher().to(device)
        model.load_state_dict(torch.load(path, map_location=device))

        # Generate Marginalized Predictions for this model
        # The scan depths are z-scores (sigmas) which match the normalized depth training
        preds = predict_teacher_marginalized(
            model, test_loader, device, Config.DEPTH_SCAN_SIGMAS
        )

        # Accumulate
        for img_id, mask in preds.items():
            if img_id not in accumulated_preds:
                accumulated_preds[img_id] = np.zeros_like(mask, dtype=np.float64)
            accumulated_preds[img_id] += mask

        counts += 1

        # Cleanup
        del model
        torch.cuda.empty_cache()

    # Average predictions
    print(f"Averaging predictions from {counts} models...")
    soft_masks = {}
    for img_id, total_mask in accumulated_preds.items():
        soft_masks[img_id] = (total_mask / counts).astype(np.float32)

    return soft_masks


def run_stage3_student(soft_masks):
    """
    Stage 3: Train Generalist Student with Multi-Task Distillation.
    Trains the student on combined labeled (Train) and unlabeled (Test+SoftLabels) data.

    Args:
        soft_masks (dict): Soft pseudo-labels for the test set.
    """
    print("\n=== Stage 3: Training Generalist Student ===")
    device = Config.DEVICE

    # Load Data
    # Labeled Data
    train_loader, val_loader, _ = get_train_val_loaders()

    # Unlabeled Data (Pseudo-labels)
    pseudo_loader = get_pseudo_loader(soft_masks)

    # Initialize Student
    model = GeneralistStudent().to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS_STUDENT, eta_min=1e-6)

    save_path = os.path.join(Config.CHECKPOINT_DIR, "student_best.pth")

    # Train Student
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=Config.EPOCHS_STUDENT,
        save_path=save_path,
        is_teacher=False,
        unlabeled_loader=pseudo_loader,
    )

    # Generate Submission
    print("\n=== Generating Final Submission ===")
    test_loader = get_test_loader()
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    generate_submission(
        model=model,
        test_loader=test_loader,
        val_loader=val_loader,
        device=device,
        output_path=submission_path,
    )
