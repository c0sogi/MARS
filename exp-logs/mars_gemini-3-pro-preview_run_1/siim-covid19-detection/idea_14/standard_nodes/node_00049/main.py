import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.cuda import amp
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import cfg
from library.dataset import SIIMDataset, process_and_cache_data
from library.model import ResNet18D_UNet
from library.loss import CompositeLoss
from library.train import train_one_epoch, valid_one_epoch
from library.utils import seed_everything, calculate_map, AverageMeter
from library.inference import run_inference


def analyze_failures(model, loader, criterion, device):
    """
    Performs failure analysis on the validation set.
    Calculates per-sample loss and correlates it with metadata features.
    """
    model.eval()

    error_magnitudes = []
    study_labels = []
    num_boxes_list = []

    # For mAP calculation
    all_preds = []
    all_targets = []

    print("Running Failure Analysis on Validation Set...")

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            cls_targets = targets["study_label"].to(device)
            seg_targets = targets["mask"].to(device)

            # Forward pass
            cls_logits, seg_logits = model(images)

            # Calculate per-sample loss (Error Magnitude)
            # We compute loss without reduction to get per-sample values
            ce_loss = torch.nn.functional.cross_entropy(
                cls_logits, cls_targets, reduction="none"
            )

            # BCE loss is usually (B, 1, H, W), we average over spatial dims to get per-sample scalar
            bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                seg_logits, seg_targets, reduction="none"
            )
            bce_loss = bce_loss.mean(dim=(1, 2, 3))

            # Composite error
            total_error = (cfg.study_loss_weight * ce_loss) + (
                cfg.image_loss_weight * bce_loss
            )

            # Store data
            error_magnitudes.extend(total_error.cpu().tolist())
            study_labels.extend(cls_targets.cpu().tolist())

            # Count boxes per sample
            for i in range(images.size(0)):
                n_boxes = targets["boxes"][i].shape[0]
                num_boxes_list.append(n_boxes)

            # --- Data Collection for mAP ---
            # Similar logic to valid_one_epoch but we need to ensure we capture everything for the final metric
            cls_preds = torch.argmax(cls_logits, dim=1)
            seg_probs = torch.sigmoid(seg_logits).cpu().numpy()

            batch_size = images.size(0)
            for b in range(batch_size):
                # Targets
                gt_boxes = targets["boxes"][b]
                gt_labels = torch.zeros((gt_boxes.shape[0],), dtype=torch.int64)
                all_targets.append({"boxes": gt_boxes, "labels": gt_labels})

                # Preds (with Gating)
                pred_study_cls = cls_preds[b].item()
                if pred_study_cls == 0:
                    p_boxes = torch.zeros((0, 4), dtype=torch.float32)
                    p_scores = torch.zeros((0,), dtype=torch.float32)
                    p_labels = torch.zeros((0,), dtype=torch.int64)
                else:
                    # Extract boxes (importing function locally to avoid circular dependency issues if any,
                    # though it's in library.train)
                    from library.train import extract_boxes_from_prob

                    raw_boxes, raw_scores = extract_boxes_from_prob(
                        seg_probs[b, 0], threshold=0.5
                    )

                    if len(raw_boxes) > 0:
                        p_boxes = torch.tensor(raw_boxes, dtype=torch.float32)
                        p_scores = torch.tensor(raw_scores, dtype=torch.float32)
                        p_labels = torch.zeros((len(raw_boxes),), dtype=torch.int64)
                    else:
                        p_boxes = torch.zeros((0, 4), dtype=torch.float32)
                        p_scores = torch.zeros((0,), dtype=torch.float32)
                        p_labels = torch.zeros((0,), dtype=torch.int64)

                all_preds.append(
                    {"boxes": p_boxes, "scores": p_scores, "labels": p_labels}
                )

    # 1. Calculate Correlations
    df_analysis = pd.DataFrame(
        {
            "error": error_magnitudes,
            "study_label": study_labels,
            "num_boxes": num_boxes_list,
        }
    )

    print("\nFailure Analysis - Feature Correlations with Error Magnitude:")
    correlations = df_analysis.corr()["error"].drop("error")
    print(correlations)

    # 2. Calculate Final mAP
    final_map = calculate_map(
        all_preds, all_targets, iou_threshold=cfg.iou_threshold, num_classes=1
    )

    return final_map


def main():
    # 1. Setup
    seed_everything(cfg.seed)

    # Override Config for Fast Baseline
    cfg.epochs = 5
    print(f"Configuration: Epochs set to {cfg.epochs} for fast baseline.")

    # 2. Data Loading
    print("Loading Data...")
    train_df, train_imgs, train_masks, train_dims = process_and_cache_data(
        "train", load_cached_data=True
    )
    val_df, val_imgs, val_masks, val_dims = process_and_cache_data(
        "val", load_cached_data=True
    )

    train_dataset = SIIMDataset(
        train_df, train_imgs, train_masks, train_dims, split="train"
    )
    val_dataset = SIIMDataset(val_df, val_imgs, val_masks, val_dims, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = ResNet18D_UNet().to(cfg.device)
    criterion = CompositeLoss().to(cfg.device)
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr
    )
    scaler = amp.GradScaler()

    # 4. Training Loop
    best_score = 0.0

    print("Starting Training...")
    for epoch in range(cfg.epochs):
        # Train
        train_loss = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            scaler,
            cfg.device,
            epoch,
        )

        # Validate
        val_loss, val_acc, val_map = valid_one_epoch(
            val_loader, model, criterion, cfg.device
        )

        # Composite Score
        composite_score = (val_acc + val_map) / 2.0
        print(
            f"Epoch {epoch+1} Summary: Loss {val_loss:.4f} | Acc {val_acc:.4f} | mAP {val_map:.4f} | Score {composite_score:.4f}"
        )

        if composite_score > best_score:
            print(f"New Best Score: {composite_score:.4f}. Saving model...")
            best_score = composite_score
            torch.save(model.state_dict(), cfg.model_save_path)

    print("Training Complete.")

    # 5. Final Evaluation & Failure Analysis
    print("\nLoading best model for final evaluation...")
    if os.path.exists(cfg.model_save_path):
        model.load_state_dict(torch.load(cfg.model_save_path, map_location=cfg.device))
    else:
        print("Warning: No model checkpoint found. Using current model state.")

    final_map = analyze_failures(model, val_loader, criterion, cfg.device)

    # REQUIRED PRINT FORMAT
    print(f"Final Validation Metric: {final_map}")

    # 6. Submission Logic
    threshold = 0.49944536565378
    if final_map > threshold:
        print(
            f"Metric ({final_map}) > Threshold ({threshold}). Generating submission..."
        )
        run_inference()
    else:
        print(f"Metric ({final_map}) <= Threshold ({threshold}). Skipping submission.")


if __name__ == "__main__":
    main()
