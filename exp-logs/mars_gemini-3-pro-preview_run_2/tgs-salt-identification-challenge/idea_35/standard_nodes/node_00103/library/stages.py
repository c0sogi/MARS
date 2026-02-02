import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import prepare_data, SaltDataset, get_transforms
from library.models import SaltNet
from library.losses import LovaszHingeLoss, StudentLoss, BCELovaszLoss
from library.engine import (
    train_one_epoch,
    validate,
    optimize_threshold,
    generate_submission,
)
from library.utils import set_seed

# Ensure cache directory exists
os.makedirs(Config.CACHE_DIR, exist_ok=True)


def run_stage_1_teacher_ensemble(data_containers, device):
    """
    Stage 1: Train a 5-Fold Ensemble of Specialist Teachers (Depth-Injected).

    Args:
        data_containers (dict): Data loaded from prepare_data.
        device (torch.device): Computation device.

    Returns:
        list: Paths to the saved teacher models that passed the gating threshold.
    """
    print("\n" + "=" * 40)
    print("STAGE 1: Teacher Ensemble Training")
    print("=" * 40)

    # Combine Train and Val data for K-Fold
    train_imgs = data_containers["train"]["images"]
    train_masks = data_containers["train"]["masks"]
    train_depths = data_containers["train"]["depths"]

    val_imgs = data_containers["val"]["images"]
    val_masks = data_containers["val"]["masks"]
    val_depths = data_containers["val"]["depths"]

    all_images = np.concatenate([train_imgs, val_imgs], axis=0)
    all_masks = np.concatenate([train_masks, val_masks], axis=0)
    all_depths = np.concatenate([train_depths, val_depths], axis=0)

    kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)

    teacher_paths = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_images)):
        print(f"\nTraining Teacher Fold {fold + 1}/{Config.N_FOLDS}")

        # Fold Data
        fold_train_imgs = all_images[train_idx]
        fold_train_masks = all_masks[train_idx]
        fold_train_depths = all_depths[train_idx]

        fold_val_imgs = all_images[val_idx]
        fold_val_masks = all_masks[val_idx]
        fold_val_depths = all_depths[val_idx]

        # Datasets
        train_ds = SaltDataset(
            fold_train_imgs,
            masks=fold_train_masks,
            depths=fold_train_depths,
            transform=get_transforms("train"),
            mode="train",
        )
        val_ds = SaltDataset(
            fold_val_imgs,
            masks=fold_val_masks,
            depths=fold_val_depths,
            transform=get_transforms("val"),
            mode="val",
        )

        # Loaders
        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Model setup
        model = SaltNet(use_depth=True, aux_head=False, pretrained=True).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Config.EPOCHS_STAGE1
        )
        criterion = BCELovaszLoss()

        # Training Loop
        best_map = 0.0
        best_model_path = os.path.join(Config.CACHE_DIR, f"teacher_fold_{fold}.pth")

        for epoch in range(Config.EPOCHS_STAGE1):
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                epoch + 1,
                criterion,
                is_student=False,
                is_pseudo=False,
            )
            val_map = validate(model, val_loader, device, is_student=False)

            scheduler.step()

            if val_map > best_map:
                best_map = val_map
                torch.save(model.state_dict(), best_model_path)
                print(f"New Best Teacher Fold {fold}: {best_map}")

        # Gating check
        if best_map >= 0.75:
            teacher_paths.append(best_model_path)
            print(f"Fold {fold} accepted with mAP {best_map}")
        else:
            print(f"Fold {fold} rejected (mAP {best_map} < 0.75)")

    return teacher_paths


def run_stage_2_marginalization(teacher_paths, data_containers, device):
    """
    Stage 2: Generate Marginalized Soft Pseudo-Labels for Test Set.

    Args:
        teacher_paths (list): Paths to trained teacher models.
        data_containers (dict): Data loaded from prepare_data.
        device (torch.device): Computation device.

    Returns:
        np.ndarray: Soft probability masks for the test set.
    """
    print("\n" + "=" * 40)
    print("STAGE 2: Marginalized Depth Scan")
    print("=" * 40)

    test_images = data_containers["test"]["images"]
    test_ids = data_containers["test"]["ids"]
    num_test = len(test_images)

    # Output accumulator: (N, 1, 128, 128) - working in padded space
    accumulated_probs = np.zeros(
        (num_test, 1, Config.IMG_TARGET_SIZE, Config.IMG_TARGET_SIZE), dtype=np.float32
    )

    scan_depths = Config.DEPTH_SCAN_RANGE
    total_scans = len(teacher_paths) * len(scan_depths)

    print(f"Scanning {len(teacher_paths)} models across {len(scan_depths)} depths...")

    for model_path in teacher_paths:
        model = SaltNet(use_depth=True, aux_head=False, pretrained=False).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        for z_val in scan_depths:
            # Create a depth array filled with z_val
            # z_val is already normalized (std dev units)
            current_depths = np.full((num_test,), z_val, dtype=np.float32)

            # Create dataset/loader
            ds = SaltDataset(
                test_images,
                masks=None,
                depths=current_depths,
                ids=test_ids,
                transform=get_transforms("test"),  # No aug
                mode="test",
            )
            loader = DataLoader(
                ds,
                batch_size=Config.BATCH_SIZE * 2,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
            )

            # Inference
            probs_list = []
            with torch.no_grad():
                for data in loader:
                    imgs = data["image"].to(device)
                    d = data["depth"].to(device)

                    logits = model(imgs, depth=d)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    probs_list.append(probs)

            # Concatenate and accumulate
            batch_probs = np.concatenate(probs_list, axis=0)
            accumulated_probs += batch_probs

    # Average
    avg_probs = accumulated_probs / total_scans

    # Save soft masks
    save_path = os.path.join(Config.CACHE_DIR, "test_soft_masks.npy")
    np.save(save_path, avg_probs)
    print(f"Soft masks saved to {save_path}")

    return avg_probs


def run_stage_3_student_distillation(soft_masks, data_containers, device):
    """
    Stage 3: Train Generalist Student on Combined Data (Labeled + Pseudo-Labeled).

    Args:
        soft_masks (np.ndarray): Soft pseudo-labels from Stage 2.
        data_containers (dict): Data loaded from prepare_data.
        device (torch.device): Computation device.

    Returns:
        str: Path to the best student model.
    """
    print("\n" + "=" * 40)
    print("STAGE 3: Student Distillation")
    print("=" * 40)

    # 1. Prepare Labeled Data (Train + Val)
    train_imgs = data_containers["train"]["images"]
    train_masks = data_containers["train"]["masks"]
    train_depths = data_containers["train"]["depths"]

    val_imgs = data_containers["val"]["images"]
    val_masks = data_containers["val"]["masks"]
    val_depths = data_containers["val"]["depths"]

    labeled_imgs = np.concatenate([train_imgs, val_imgs], axis=0)
    labeled_masks = np.concatenate([train_masks, val_masks], axis=0)
    labeled_depths = np.concatenate([train_depths, val_depths], axis=0)

    # 2. Prepare Unlabeled Data (Test + Soft Masks)
    test_imgs = data_containers["test"]["images"]
    # soft_masks shape: (N, 1, 128, 128). SaltDataset expects (N, H, W) or (N, H, W, 1)
    unlabeled_masks = soft_masks.squeeze(1)

    # Dummy depths for unlabeled (won't be used by loss in is_pseudo=True mode)
    unlabeled_depths = np.zeros((len(test_imgs),), dtype=np.float32)

    # 3. Datasets & Loaders
    labeled_ds = SaltDataset(
        labeled_imgs,
        masks=labeled_masks,
        depths=labeled_depths,
        transform=get_transforms("train"),
        mode="train",
    )

    unlabeled_ds = SaltDataset(
        test_imgs,
        masks=unlabeled_masks,
        depths=unlabeled_depths,
        transform=get_transforms("train"),  # Augment pseudo-labeled data too
        mode="train",
    )

    # Validation Set (Use original Val set for metric tracking)
    val_ds = SaltDataset(
        val_imgs,
        masks=val_masks,
        depths=val_depths,
        transform=get_transforms("val"),
        mode="val",
    )

    labeled_loader = DataLoader(
        labeled_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        drop_last=True,
    )
    unlabeled_loader = DataLoader(
        unlabeled_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # 4. Model Setup
    model = SaltNet(use_depth=False, aux_head=True, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS_STAGE3
    )
    criterion = StudentLoss()

    # 5. Training Loop
    best_map = 0.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_student_model.pth")

    for epoch in range(Config.EPOCHS_STAGE3):
        print(f"Epoch {epoch+1}/{Config.EPOCHS_STAGE3}")

        # Train Labeled
        loss_lab = train_one_epoch(
            model,
            labeled_loader,
            optimizer,
            device,
            epoch + 1,
            criterion,
            is_student=True,
            is_pseudo=False,
        )

        # Train Unlabeled
        loss_unlab = train_one_epoch(
            model,
            unlabeled_loader,
            optimizer,
            device,
            epoch + 1,
            criterion,
            is_student=True,
            is_pseudo=True,
        )

        print(f"  Labeled Loss: {loss_lab:.4f} | Unlabeled Loss: {loss_unlab:.4f}")

        # Validate
        val_map = validate(model, val_loader, device, is_student=True)

        scheduler.step()

        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), best_model_path)
            print(f"  New Best Student: {best_map}")

    return best_model_path


def run_pipeline():
    """
    Main execution pipeline for the Marginalized-Scan Multi-Task Distillation strategy.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Load Data
    data_containers = prepare_data(load_cached_data=True)

    # 2. Stage 1: Teacher Ensemble
    teacher_paths = run_stage_1_teacher_ensemble(data_containers, device)

    if not teacher_paths:
        print("No teacher models reached the threshold. Aborting.")
        return

    # 3. Stage 2: Marginalization
    soft_masks = run_stage_2_marginalization(teacher_paths, data_containers, device)

    # 4. Stage 3: Student Distillation
    student_model_path = run_stage_3_student_distillation(
        soft_masks, data_containers, device
    )

    # 5. Final Inference & Submission
    print("\n" + "=" * 40)
    print("Generating Submission")
    print("=" * 40)

    # Load best student
    student_model = SaltNet(use_depth=False, aux_head=True, pretrained=False).to(device)
    student_model.load_state_dict(torch.load(student_model_path, map_location=device))

    # Optimize Threshold on Validation Set
    val_ds = SaltDataset(
        data_containers["val"]["images"],
        masks=data_containers["val"]["masks"],
        depths=data_containers["val"][
            "depths"
        ],  # Not used by student but required by dataset
        transform=get_transforms("val"),
        mode="val",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    best_threshold = optimize_threshold(student_model, val_loader, device)

    # Generate Test Predictions
    test_ds = SaltDataset(
        data_containers["test"]["images"],
        ids=data_containers["test"]["ids"],
        transform=get_transforms("test"),
        mode="test",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    generate_submission(student_model, test_loader, device, threshold=best_threshold)
