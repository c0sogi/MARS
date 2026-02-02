import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from sklearn.model_selection import KFold
import copy
import cv2

from library.config import Config, seed_everything
from library.utils import unpad_image, pad_image, rle_encode
from library.dataset import process_data, SaltDataset, get_transforms, get_loaders
from library.models import TeacherLinkNet, StudentLinkNet
from library.training import train_model, generate_submission
from library.losses import StudentLoss, LovaszHingeLoss

# Ensure reproducibility
seed_everything(Config.SEED)


def run_stage1_teacher(debug=False):
    """
    Stage 1: Train the Privileged Teacher Model using Image + Depth.
    """
    print("=" * 50)
    print("STAGE 1: Training Privileged Teacher")
    print("=" * 50)

    # Use provided data loader helper which handles splitting and normalization
    # Note: get_loaders handles caching internally
    train_loader, val_loader, _ = get_loaders(load_cached_data=True)

    if debug:
        print("DEBUG Mode: Reducing epochs for Teacher")
        Config.NUM_EPOCHS_TEACHER = 2

    # Initialize Teacher Model
    model = TeacherLinkNet(num_classes=1).to(Config.DEVICE)

    # Train Model
    # Teacher uses standard Lovasz+BCE loss (handled inside train_model when is_student=False)
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=Config.DEVICE,
        config=Config,
        teacher_model=None,
        is_student=False,
    )

    # Save final stage 1 model
    save_path = os.path.join(Config.CACHE_DIR, "stage1_teacher.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Stage 1 complete. Teacher model saved to {save_path}")

    return save_path


def run_stage2_student_ensemble(teacher_path, debug=False):
    """
    Stage 2: Train Multi-Task Student Ensemble (5-Fold) with Distillation.
    """
    print("=" * 50)
    print("STAGE 2: Training Student Ensemble (Distillation)")
    print("=" * 50)

    if debug:
        print("DEBUG Mode: Reducing epochs for Student")
        Config.NUM_EPOCHS_STUDENT = 2

    # 1. Load Teacher Model
    teacher_model = TeacherLinkNet(num_classes=1).to(Config.DEVICE)
    teacher_model.load_state_dict(torch.load(teacher_path, map_location=Config.DEVICE))
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # 2. Prepare Data for K-Fold (Combine Train + Val from metadata)
    # We need to manually load and combine to perform our own K-Fold split
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)

    # Process data (loads from cache if available)
    t_imgs, t_masks, t_depths, t_ids = process_data(train_df, "train", Config.CACHE_DIR)
    v_imgs, v_masks, v_depths, v_ids = process_data(val_df, "val", Config.CACHE_DIR)

    # Combine
    all_images = np.concatenate([t_imgs, v_imgs], axis=0)
    all_masks = np.concatenate([t_masks, v_masks], axis=0)
    all_depths = np.concatenate([t_depths, v_depths], axis=0)
    all_ids = np.concatenate([t_ids, v_ids], axis=0)

    # Normalize Depths (Global Stats)
    d_mean = np.mean(all_depths)
    d_std = np.std(all_depths) + 1e-8
    all_depths_norm = (all_depths - d_mean) / d_std

    # K-Fold Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=Config.SEED)

    fold_model_paths = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_images)):
        print(f"\n--- Fold {fold + 1}/5 ---")

        # Create Datasets
        train_ds = SaltDataset(
            all_images[train_idx],
            all_depths_norm[train_idx],
            all_ids[train_idx],
            all_masks[train_idx],
            transform=get_transforms("train"),
        )
        val_ds = SaltDataset(
            all_images[val_idx],
            all_depths_norm[val_idx],
            all_ids[val_idx],
            all_masks[val_idx],
            transform=get_transforms("val"),
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

        # Initialize Student
        student_model = StudentLinkNet(num_classes=1).to(Config.DEVICE)

        # Train with Distillation
        # We pass the teacher_model to train_model, which triggers the distillation loss in StudentLoss
        student_model = train_model(
            model=student_model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=Config.DEVICE,
            config=Config,
            teacher_model=teacher_model,
            is_student=True,
        )

        # Validate and Check Gating (mAP > 0.70)
        # We re-run validation to get the exact metric
        trainer_temp = from_model_trainer(
            student_model
        )  # Helper to reuse validation logic
        val_map, _ = trainer_temp.validate(val_loader)

        print(f"Fold {fold+1} Final mAP: {val_map:.6f}")

        if val_map >= 0.70:
            save_name = f"stage2_fold{fold}.pth"
            save_path = os.path.join(Config.CACHE_DIR, save_name)
            torch.save(student_model.state_dict(), save_path)
            fold_model_paths.append(save_path)
            print(f"Model saved: {save_path}")
        else:
            print(f"Model discarded (mAP < 0.70)")

        if debug:
            break  # Run only one fold in debug mode

    return fold_model_paths


class PseudoLabelDataset(Dataset):
    """Dataset for Unlabeled data with Soft Pseudo Labels."""

    def __init__(self, images, ids, soft_masks, transform=None):
        self.images = images
        self.ids = ids
        self.soft_masks = soft_masks  # (N, 128, 128) float
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.soft_masks[idx]

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        if self.transform:
            # Albumentations expects mask to be numpy
            # Note: Albumentations might cast float masks if not careful, but usually preserves type if not standardizing
            # We manually handle normalization for image
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]  # returns numpy

        # Convert mask to tensor (Float for BCE with soft targets)
        mask = torch.from_numpy(mask).float()
        # Add channel dim: (H, W) -> (1, H, W)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return {"image": image, "mask": mask, "id": self.ids[idx]}  # Soft labels


def run_stage3_self_training(student_paths, debug=False):
    """
    Stage 3: Self-Training with Noisy Student.
    Generates pseudo-labels on Test set and retrains on Combined (Train+Test) dataset.
    """
    print("=" * 50)
    print("STAGE 3: Self-Training (Noisy Student)")
    print("=" * 50)

    if not student_paths:
        print("No valid student models from Stage 2. Skipping Stage 3.")
        return

    if debug:
        Config.NUM_EPOCHS_FINAL = 2

    # 1. Load Test Data
    test_df = pd.read_csv(Config.TEST_CSV)
    test_imgs, _, test_depths_raw, test_ids = process_data(
        test_df, "test", Config.CACHE_DIR
    )

    # Normalize test depths using stats from Stage 2 (re-calculate or load)
    # For simplicity, we recalculate on training data to ensure consistency
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    _, _, t_depths, _ = process_data(train_df, "train", Config.CACHE_DIR)
    _, _, v_depths, _ = process_data(val_df, "val", Config.CACHE_DIR)
    all_depths = np.concatenate([t_depths, v_depths])
    d_mean, d_std = np.mean(all_depths), np.std(all_depths) + 1e-8

    # Test dataset for inference
    test_ds = SaltDataset(
        test_imgs,
        (test_depths_raw - d_mean) / d_std,
        test_ids,
        masks=None,
        transform=get_transforms("test"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # 2. Generate Soft Pseudo Labels (Ensemble + TTA)
    print("Generating Pseudo Labels...")
    ensemble_models = []
    for path in student_paths:
        m = StudentLinkNet(num_classes=1).to(Config.DEVICE)
        m.load_state_dict(torch.load(path, map_location=Config.DEVICE))
        m.eval()
        ensemble_models.append(m)

    soft_preds = np.zeros(
        (len(test_imgs), Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images = batch["image"].to(Config.DEVICE).float()
            bs = images.size(0)

            batch_preds = torch.zeros(
                (bs, 1, Config.IMG_SIZE, Config.IMG_SIZE), device=Config.DEVICE
            )

            for model in ensemble_models:
                # Original
                logits = model(images)["logits"]
                probs = torch.sigmoid(logits)

                # TTA: Horizontal Flip
                images_flip = torch.flip(images, dims=[3])
                logits_flip = model(images_flip)["logits"]
                probs_flip = torch.flip(torch.sigmoid(logits_flip), dims=[3])

                batch_preds += (probs + probs_flip) / 2.0

            batch_preds /= len(ensemble_models)

            # Store
            start_idx = batch_idx * Config.BATCH_SIZE
            end_idx = start_idx + bs
            soft_preds[start_idx:end_idx] = batch_preds.squeeze(1).cpu().numpy()

    # 3. Prepare Combined Datasets
    # Labeled Data
    t_imgs, t_masks, t_depths, t_ids = process_data(train_df, "train", Config.CACHE_DIR)
    v_imgs, v_masks, v_depths, v_ids = process_data(val_df, "val", Config.CACHE_DIR)

    labeled_imgs = np.concatenate([t_imgs, v_imgs])
    labeled_masks = np.concatenate([t_masks, v_masks])
    labeled_depths = np.concatenate([t_depths, v_depths])
    labeled_depths = (labeled_depths - d_mean) / d_std
    labeled_ids = np.concatenate([t_ids, v_ids])

    # We use a subset of labeled data as validation for the final stage (e.g., original val set)
    # But to maximize performance, we train on ALL labeled + pseudo-labeled test.
    # We will just use the original validation set to monitor progress.

    labeled_ds = SaltDataset(
        labeled_imgs,
        labeled_depths,
        labeled_ids,
        labeled_masks,
        transform=get_transforms("train"),
    )
    unlabeled_ds = PseudoLabelDataset(
        test_imgs, test_ids, soft_preds, transform=get_transforms("train")
    )

    # Validation set (Original Val)
    val_ds_final = SaltDataset(
        v_imgs,
        (v_depths - d_mean) / d_std,
        v_ids,
        v_masks,
        transform=get_transforms("val"),
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
        val_ds_final,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # 4. Train Final Student
    final_model = StudentLinkNet(num_classes=1).to(Config.DEVICE)
    optimizer = torch.optim.AdamW(
        final_model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS_FINAL
    )

    # Losses
    student_loss_fn = StudentLoss()  # Handles Labeled (Seg + MSE)
    bce_loss = nn.BCEWithLogitsLoss()  # Handles Unlabeled (Soft Labels)

    best_map = 0.0
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model_stage3.pth")

    print("Starting Final Training Loop...")

    for epoch in range(Config.NUM_EPOCHS_FINAL):
        final_model.train()
        train_loss = 0.0

        # Iterate over zipped loaders (limit to length of smaller one, or cycle larger?
        # Usually labeled is larger (3000) than test (1000). We iterate labeled and cycle unlabeled)

        # Helper to cycle unlabeled
        def cycle(iterable):
            while True:
                for x in iterable:
                    yield x

        unlabeled_iter = cycle(unlabeled_loader)

        for batch_labeled in labeled_loader:
            batch_unlabeled = next(unlabeled_iter)

            optimizer.zero_grad()

            # --- Labeled Step ---
            l_imgs = batch_labeled["image"].to(Config.DEVICE).float()
            l_masks = batch_labeled["mask"].to(Config.DEVICE).float()
            l_depths = batch_labeled["depth"].to(Config.DEVICE).float()

            l_out = final_model(l_imgs)
            # Loss: Seg + MSE (No distillation from teacher here, or we could, but prompt says BCE vs Pseudo)
            # We pass teacher_logits=None
            loss_labeled, _ = student_loss_fn(
                l_out, l_masks, l_depths, teacher_logits=None
            )

            # --- Unlabeled Step ---
            u_imgs = batch_unlabeled["image"].to(Config.DEVICE).float()
            u_masks = batch_unlabeled["mask"].to(Config.DEVICE).float()  # Soft targets

            u_out = final_model(u_imgs)
            u_logits = u_out["logits"]

            # Loss: BCE against soft targets
            loss_unlabeled = bce_loss(u_logits, u_masks)

            # Combine
            loss = loss_labeled + loss_unlabeled

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(labeled_loader)

        # Validation
        # Use a temporary trainer helper to reuse validate logic
        trainer_temp = from_model_trainer(final_model)
        val_map, val_thresh = trainer_temp.validate(val_loader)

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS_FINAL} | Loss: {train_loss:.6f} | Val mAP: {val_map:.8f}"
        )

        if val_map > best_map:
            best_map = val_map
            torch.save(final_model.state_dict(), best_model_path)
            # Save threshold
            with open(os.path.join(Config.CACHE_DIR, "best_threshold.txt"), "w") as f:
                f.write(str(val_thresh))

    # 5. Generate Final Submission
    print("Generating Final Submission...")
    final_model.load_state_dict(torch.load(best_model_path))

    # We need the test loader again (standard one)
    test_loader_final, _, _ = get_loaders(
        load_cached_data=True
    )  # This returns (train, val, test)
    # Actually get_loaders returns (train, val, test). We just need the 3rd one.
    _, _, test_loader_std = get_loaders(load_cached_data=True)

    generate_submission(
        final_model, test_loader_std, Config.DEVICE, Config.SUBMISSION_PATH
    )


# Helper to instantiate a Trainer just for validation reuse
def from_model_trainer(model):
    from library.training import Trainer

    return Trainer(model, Config.DEVICE)
