import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cv2
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from sklearn.metrics import accuracy_score

# Import provided library modules
from library.config import CFG, seed_everything
from library.utils import AverageMeter, accuracy
from library.dataset import CassavaDataset, get_transforms, Mixup
from library.model import CassavaViT
from library.engine import train_one_epoch, valid_one_epoch
from library.inference import run_inference


def main():
    # 1. Setup
    seed_everything(CFG.seed)
    device = torch.device(CFG.device)
    print(f"Using device: {device}")

    # 2. Data Loading
    if not os.path.exists(CFG.train_csv) or not os.path.exists(CFG.val_csv):
        print("Error: Metadata files not found.")
        return

    train_df = pd.read_csv(CFG.train_csv)
    val_df = pd.read_csv(CFG.val_csv)

    # Create Datasets
    train_dataset = CassavaDataset(train_df, transform=get_transforms("train"))
    val_dataset = CassavaDataset(val_df, transform=get_transforms("valid"))

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    # 3. Model Initialization
    print(f"Initializing model: {CFG.model_name}")
    model = CassavaViT(
        model_name=CFG.model_name,
        pretrained=CFG.pretrained,
        num_classes=CFG.num_classes,
    )
    model.to(device)

    # 4. Training Configuration
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.epochs, eta_min=CFG.min_lr
    )

    # CrossEntropyLoss handles both indices and soft probabilities (from Mixup)
    criterion = nn.CrossEntropyLoss(label_smoothing=CFG.label_smoothing)

    mixup_fn = Mixup(
        mixup_alpha=CFG.mixup_alpha,
        cutmix_alpha=CFG.cutmix_alpha,
        prob=CFG.mixup_prob,
        num_classes=CFG.num_classes,
    )

    # 5. Training Loop
    best_acc = 0.0
    best_model_path = os.path.join(CFG.output_dir, "best_model.pth")

    # SWA Initialization
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=CFG.swa_lr)

    print(f"Starting training for {CFG.epochs} epochs...")
    for epoch in range(CFG.epochs):
        # Train Step
        train_loss, train_acc = train_one_epoch(
            epoch, model, train_loader, optimizer, criterion, device, mixup_fn=mixup_fn
        )

        if epoch < CFG.swa_start:
            # Normal Validation Step
            val_loss, val_acc = valid_one_epoch(
                epoch, model, val_loader, criterion, device
            )

            # Update Standard Scheduler
            scheduler.step()

            print(
                f"Epoch {epoch+1}/{CFG.epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
            )

            # Save Best Model (Standard)
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved best model with Acc: {best_acc:.4f}")
        else:
            # SWA Step
            swa_model.update_parameters(model)
            swa_scheduler.step()
            print(
                f"Epoch {epoch+1}/{CFG.epochs} | SWA Update | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}"
            )

    # SWA Finalization
    print("Updating SWA Batch Normalization statistics...")
    update_bn(train_loader, swa_model, device=device)

    # Evaluate SWA Model
    print("Evaluating SWA model...")
    swa_val_loss, swa_val_acc = valid_one_epoch(
        CFG.epochs, swa_model, val_loader, criterion, device
    )
    print(f"SWA Final Acc: {swa_val_acc:.4f} (Best Standard Acc: {best_acc:.4f})")

    if swa_val_acc > best_acc:
        print("SWA model performed better. Saving SWA model as best model.")
        torch.save(swa_model.module.state_dict(), best_model_path)
    else:
        print("Standard model performed better. Keeping best standard model.")

    # 6. Final Validation & Failure Analysis
    print("\nLoading best model for analysis...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    val_preds = []
    val_targets = []

    # Run inference on validation set
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_targets.extend(labels.numpy())

    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)

    final_metric = accuracy_score(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")
    errors = (val_preds != val_targets).astype(int)

    print("Collecting metadata features for correlation analysis...")
    file_sizes = []
    mean_brightness = []

    # Calculate simple image stats for correlation
    for path in val_df["file_path"]:
        full_path = os.path.join(CFG.input_root, path)
        try:
            # File size
            file_sizes.append(os.path.getsize(full_path))

            # Brightness (read image)
            img = cv2.imread(full_path)
            if img is not None:
                mean_brightness.append(np.mean(img))
            else:
                mean_brightness.append(0)
        except Exception:
            file_sizes.append(0)
            mean_brightness.append(0)

    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "true_label": val_targets,
            "file_size": file_sizes,
            "brightness": mean_brightness,
        }
    )

    # Calculate correlation
    correlations = analysis_df.corr()["error"]
    print("Correlation between Error and Features:")
    print(correlations)

    # 7. Submission
    THRESHOLD = 0.8889185580774366
    if final_metric > THRESHOLD:
        print(f"\nMetric {final_metric} > {THRESHOLD}. Generating submission...")
        run_inference(best_model_path, "./submission/submission.csv")
    else:
        print(f"\nMetric {final_metric} <= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
