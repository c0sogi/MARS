import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed, save_checkpoint, load_checkpoint
from library.dataset import get_dataloaders
from library.model import build_model
from library.train import train_one_epoch, validate
from library.predict import inference_with_tta


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Cite solution_lesson_node_00007: Use extended schedule (30 epochs) for better convergence
    Config.epochs = 30

    # Set device
    Config.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(Config.device)

    # Reproducibility
    set_seed(Config.seed)

    print(f"Starting run on device: {device}")
    print(
        f"Configuration: {Config.epochs} epochs, {Config.warmup_epochs} warmup epoch(s)"
    )

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        load_cached_data=True
    )

    # -------------------------------------------------------------------------
    # 3. Model Construction
    # -------------------------------------------------------------------------
    model = build_model(num_classes=Config.num_classes, pretrained=Config.pretrained)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # -------------------------------------------------------------------------
    # 4. Training Pipeline
    # -------------------------------------------------------------------------

    # --- Phase 1: Warm-up (Head Only) ---
    print("\nPhase 1: Warm-up (Head Only)")

    # Freeze backbone, unfreeze classifier
    for param in model.parameters():
        param.requires_grad = False
    for param in model.get_classifier().parameters():
        param.requires_grad = True

    optimizer_warmup = optim.AdamW(
        model.get_classifier().parameters(), lr=Config.lr_warmup
    )

    for epoch in range(1, Config.warmup_epochs + 1):
        train_one_epoch(model, train_loader, optimizer_warmup, criterion, device, epoch)
        validate(model, val_loader, criterion, device)

    # --- Phase 2: Fine-tuning (Full Model) ---
    print("\nPhase 2: Fine-tuning (Full Model)")

    # Unfreeze everything
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.lr_fine_tune, weight_decay=Config.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.epochs)

    best_metric = float("inf")

    for epoch in range(1, Config.epochs + 1):
        train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()

        # Save best model
        if val_loss < best_metric:
            best_metric = val_loss
            save_checkpoint(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_metric": best_metric,
                },
                True,
                Config.best_model_path,
            )

    # -------------------------------------------------------------------------
    # 5. Final Validation Assessment
    # -------------------------------------------------------------------------
    print("\nPerforming Final Validation Assessment...")
    # Load best model weights
    load_checkpoint(model, Config.best_model_path)
    model.eval()

    # Compute Final Metric (Log Loss)
    # validate() returns the average CrossEntropyLoss over the dataset
    final_metric = validate(model, val_loader, criterion, device)
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 6. Failure Analysis
    # -------------------------------------------------------------------------
    print("\nPerforming Failure Analysis...")
    val_dataset = val_loader.dataset
    criterion_none = nn.CrossEntropyLoss(reduction="none")
    all_losses = []

    # Compute per-sample losses
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            batch_losses = criterion_none(outputs, targets)
            all_losses.extend(batch_losses.cpu().numpy())

    # Load metadata and extract features for correlation analysis
    # val_loader is sequential (shuffle=False), so it aligns with metadata
    df_analysis = val_dataset.metadata.copy()

    # Safety check for alignment
    if len(all_losses) != len(df_analysis):
        print(
            f"Warning: Loss count ({len(all_losses)}) differs from metadata ({len(df_analysis)}). Truncating to minimum."
        )
        min_len = min(len(all_losses), len(df_analysis))
        all_losses = all_losses[:min_len]
        df_analysis = df_analysis.iloc[:min_len]

    df_analysis["loss"] = all_losses

    widths, heights, aspect_ratios, file_sizes = [], [], [], []

    # Extract image properties
    for idx, row in df_analysis.iterrows():
        full_path = os.path.join(Config.input_dir, row["file_path"])
        try:
            f_size = os.path.getsize(full_path)
            img = cv2.imread(full_path)
            if img is not None:
                h, w = img.shape[:2]
                ar = w / h if h > 0 else 0
            else:
                h, w, ar = 0, 0, 0
        except Exception:
            f_size, h, w, ar = 0, 0, 0, 0

        widths.append(w)
        heights.append(h)
        aspect_ratios.append(ar)
        file_sizes.append(f_size)

    df_analysis["width"] = widths
    df_analysis["height"] = heights
    df_analysis["aspect_ratio"] = aspect_ratios
    df_analysis["file_size"] = file_sizes

    # Calculate Correlations
    print("Correlation between Error Magnitude and Input Features:")
    # Filter out any failed image loads
    mask = df_analysis["width"] > 0
    df_clean = df_analysis[mask]

    features = ["width", "height", "aspect_ratio", "file_size"]
    for feat in features:
        if df_clean[feat].std() > 0:
            corr = np.corrcoef(df_clean["loss"], df_clean[feat])[0, 1]
            print(f"  {feat}: {corr:.6f}")
        else:
            print(f"  {feat}: NaN (No variance)")

    # -------------------------------------------------------------------------
    # 7. Submission Generation
    # -------------------------------------------------------------------------
    THRESHOLD = 0.14004325100369866

    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} < {THRESHOLD}. Generating submission...")
        df_sub = inference_with_tta(model, test_loader, device, classes)
        df_sub.to_csv(Config.submission_path, index=False)
        print(f"Submission saved to {Config.submission_path}")
    else:
        print(f"\nMetric {final_metric} >= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
