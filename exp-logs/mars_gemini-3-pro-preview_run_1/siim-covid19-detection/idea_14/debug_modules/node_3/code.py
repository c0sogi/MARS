import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from torch.cuda import amp

# Import provided library modules
from library.config import cfg
from library.utils import seed_everything, calculate_map
from library.dataset import process_and_cache_data, SIIMDataset
from library.model import ResNet18D_UNet
from library.loss import CompositeLoss
from library.train import train_one_epoch, valid_one_epoch
from library.inference import run_inference


def detection_collate(batch):
    """
    Custom collate function to handle variable-size bounding box tensors.
    """
    images = []
    masks = []
    study_labels = []
    boxes = []
    labels = []
    image_ids = []
    study_ids = []

    for img, target in batch:
        images.append(img)
        masks.append(target["mask"])
        study_labels.append(target["study_label"])
        boxes.append(target["boxes"])
        labels.append(target["labels"])
        image_ids.append(target["image_id"])
        study_ids.append(target["study_id"])

    images = torch.stack(images, 0)
    masks = torch.stack(masks, 0)
    study_labels = torch.stack(study_labels, 0)

    targets = {
        "mask": masks,
        "study_label": study_labels,
        "boxes": boxes,
        "labels": labels,
        "image_id": image_ids,
        "study_id": study_ids,
    }

    return images, targets


def run_demo():
    print("--- Starting Library Usage Demo ---")

    # =================================================================
    # 1. Configuration & Setup
    # =================================================================
    # Override config for a fast demo run
    demo_dir = "./working/demo_execution"
    os.makedirs(demo_dir, exist_ok=True)

    cfg.working_dir = demo_dir
    cfg.output_dir = demo_dir
    cfg.model_save_path = os.path.join(demo_dir, "best_model.pth")
    cfg.submission_path = os.path.join(demo_dir, "submission.csv")

    # Reduce compute requirements
    cfg.epochs = 1
    cfg.batch_size = 8
    cfg.num_workers = 2

    seed_everything(cfg.seed)
    print(f"Configuration updated. Working directory: {cfg.working_dir}")

    # =================================================================
    # 2. Data Loading & Verification
    # =================================================================
    print("\n[Step 2] Processing Data and Creating Loaders...")

    # Load and cache data (using the provided metadata)
    # This creates .npy files in cfg.working_dir
    train_df, train_imgs, train_masks, train_dims = process_and_cache_data(
        "train", load_cached_data=True
    )
    val_df, val_imgs, val_masks, val_dims = process_and_cache_data(
        "val", load_cached_data=True
    )

    # Initialize Datasets
    full_train_dataset = SIIMDataset(
        train_df, train_imgs, train_masks, train_dims, split="train"
    )
    full_val_dataset = SIIMDataset(val_df, val_imgs, val_masks, val_dims, split="val")

    # Create Subsets for speed (50 samples for train, 20 for val)
    train_indices = list(range(min(50, len(full_train_dataset))))
    val_indices = list(range(min(20, len(full_val_dataset))))

    train_subset = Subset(full_train_dataset, train_indices)
    val_subset = Subset(full_val_dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        collate_fn=detection_collate,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=detection_collate,
    )

    print(f"Created subset loaders: Train={len(train_subset)}, Val={len(val_subset)}")

    # Verify Batch Shapes
    sample_imgs, sample_targets = next(iter(train_loader))
    print(f"Batch Image Shape: {sample_imgs.shape}")
    print(f"Batch Mask Shape: {sample_targets['mask'].shape}")

    # Assertions
    assert sample_imgs.shape == (
        cfg.batch_size,
        3,
        cfg.image_size,
        cfg.image_size,
    ), "Incorrect image batch shape"
    assert sample_targets["mask"].shape == (
        cfg.batch_size,
        1,
        cfg.image_size,
        cfg.image_size,
    ), "Incorrect mask batch shape"
    assert sample_targets["study_label"].shape == (
        cfg.batch_size,
    ), "Incorrect study label shape"

    # Verify variable size targets are lists
    assert isinstance(sample_targets["boxes"], list), "Boxes should be a list"
    assert len(sample_targets["boxes"]) == cfg.batch_size, "Boxes list length mismatch"

    # =================================================================
    # 3. Model & Loss Verification
    # =================================================================
    print("\n[Step 3] Initializing Model and Loss...")

    model = ResNet18D_UNet().to(cfg.device)
    criterion = CompositeLoss().to(cfg.device)

    # Forward pass verification
    with torch.no_grad():
        dummy_input = sample_imgs.to(cfg.device)
        cls_logits, seg_logits = model(dummy_input)

    print(f"Logits Shape: {cls_logits.shape}")
    print(f"Seg Maps Shape: {seg_logits.shape}")

    assert cls_logits.shape == (
        cfg.batch_size,
        cfg.num_study_classes,
    ), "Incorrect classification logits shape"
    assert seg_logits.shape == (
        cfg.batch_size,
        cfg.num_seg_classes,
        cfg.image_size,
        cfg.image_size,
    ), "Incorrect segmentation logits shape"

    # Loss calculation verification
    cls_targets = sample_targets["study_label"].to(cfg.device)
    seg_targets = sample_targets["mask"].to(cfg.device)

    loss, loss_cls, loss_seg = criterion(
        cls_logits, seg_logits, cls_targets, seg_targets
    )
    print(
        f"Calculated Loss: {loss.item():.4f} (Study: {loss_cls.item():.4f}, Seg: {loss_seg.item():.4f})"
    )
    assert not torch.isnan(loss), "Loss is NaN"

    # =================================================================
    # 4. Training Loop Demonstration
    # =================================================================
    print("\n[Step 4] Running Training Loop (1 Epoch on Subset)...")

    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr
    )
    scaler = amp.GradScaler()

    # Train one epoch
    train_loss = train_one_epoch(
        train_loader,
        model,
        criterion,
        optimizer,
        scheduler,
        scaler,
        cfg.device,
        epoch=0,
    )

    # Validate one epoch
    val_loss, val_acc, val_map = valid_one_epoch(
        val_loader, model, criterion, cfg.device
    )

    print(
        f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mAP: {val_map:.4f}"
    )

    # Save model for inference step
    torch.save(model.state_dict(), cfg.model_save_path)
    print(f"Model saved to {cfg.model_save_path}")
    assert os.path.exists(cfg.model_save_path), "Model file was not saved"

    # =================================================================
    # 5. Metric Logic Verification
    # =================================================================
    print("\n[Step 5] Verifying mAP Metric Logic...")
    # Create synthetic perfect predictions
    # Box format: [x1, y1, x2, y2]
    mock_preds = [
        {
            "boxes": torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
            "scores": torch.tensor([0.9], dtype=torch.float32),
            "labels": torch.tensor([0], dtype=torch.int64),
        }
    ]
    mock_targets = [
        {
            "boxes": torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
            "labels": torch.tensor([0], dtype=torch.int64),
        }
    ]

    score = calculate_map(mock_preds, mock_targets, iou_threshold=0.5, num_classes=1)
    print(f"Perfect Match mAP: {score}")
    assert (
        abs(score - 1.0) < 1e-5
    ), f"Metric calculation failed for perfect match: {score}"

    # =================================================================
    # 6. Inference Demonstration
    # =================================================================
    print("\n[Step 6] Running Inference Pipeline...")

    # run_inference loads the model from cfg.model_save_path and processes the test set
    # The test set is small enough (~600 images) to run fully in a few minutes.
    run_inference()

    # Verify submission
    if os.path.exists(cfg.submission_path):
        df_sub = pd.read_csv(cfg.submission_path)
        print(f"Submission generated successfully.")
        print(df_sub.head())
        print(f"Rows: {len(df_sub)}")

        # Basic format check
        assert (
            "Id" in df_sub.columns and "PredictionString" in df_sub.columns
        ), "Submission columns missing"
        assert len(df_sub) > 0, "Submission file is empty"
    else:
        raise FileNotFoundError(f"Submission file not found at {cfg.submission_path}")

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
