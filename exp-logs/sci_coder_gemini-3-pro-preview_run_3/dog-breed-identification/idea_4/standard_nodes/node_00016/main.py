import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

from library.config import Config
from library.utils import set_seed, load_checkpoint
from library.data import get_dataloaders
from library.model import create_model, freeze_backbone, unfreeze_backbone
from library.engine import train_phase, validate, generate_submission


def analyze_failures(model, dataloader, metadata_path, device):
    """
    Performs failure analysis by correlating prediction error with input features.
    """
    print("\nStarting Failure Analysis...")
    model.eval()
    all_losses = []
    criterion = nn.CrossEntropyLoss(reduction="none")

    # Calculate per-sample loss
    # Note: dataloader must be sequential (shuffle=False) which is true for val_loader
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            all_losses.extend(loss.cpu().numpy())

    # Load metadata to get image features
    df = pd.read_csv(metadata_path)

    # Ensure lengths match
    if len(df) != len(all_losses):
        print(
            f"Warning: Metadata length ({len(df)}) does not match predictions ({len(all_losses)}). Skipping analysis."
        )
        return

    df["loss"] = all_losses

    # Extract image features
    sizes = []
    aspect_ratios = []
    widths = []
    heights = []

    print("Extracting image metadata for correlation analysis...")
    for idx, row in df.iterrows():
        path = os.path.join(Config.INPUT_DIR, row["file_path"])
        try:
            # File size
            sz = os.path.getsize(path)

            # Dimensions
            img = cv2.imread(path)
            if img is not None:
                h, w = img.shape[:2]
                ar = w / h if h > 0 else 0
            else:
                h, w, ar = 0, 0, 0
        except Exception:
            sz, h, w, ar = 0, 0, 0, 0

        sizes.append(sz)
        aspect_ratios.append(ar)
        widths.append(w)
        heights.append(h)

    df["file_size"] = sizes
    df["aspect_ratio"] = aspect_ratios
    df["width"] = widths
    df["height"] = heights

    # Calculate correlations
    print("\nFailure Analysis (Correlation with Error Magnitude):")
    features = ["file_size", "aspect_ratio", "width", "height"]
    for col in features:
        corr, p = spearmanr(df["loss"], df[col])
        print(f"  {col}: Correlation={corr:.4f}, p-value={p:.4f}")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Initialize Model
    print("Initializing model...")
    model = create_model(num_classes=Config.NUM_CLASSES, pretrained=True)
    model.to(device)

    # ==========================================
    # Phase 1: Warmup
    # ==========================================
    print("\n=== Phase 1: Warmup (Frozen Backbone) ===")
    freeze_backbone(model)

    # Loaders for Phase 1
    train_loader, val_loader, _, _ = get_dataloaders(
        Config.PHASE1_RES, Config.PHASE1_BATCH_SIZE
    )

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.PHASE1_LR,
        weight_decay=Config.WEIGHT_DECAY,
    )

    train_phase(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=None,
        num_epochs=Config.PHASE1_EPOCHS,
        device=device,
        patience=Config.EARLY_STOPPING_PATIENCE,
        save_name="checkpoint_phase1.pth",
    )

    # ==========================================
    # Phase 2: Standard Resolution Training
    # ==========================================
    print("\n=== Phase 2: Standard Resolution Training (Unfrozen) ===")
    unfreeze_backbone(model)

    # Loaders for Phase 2
    train_loader, val_loader, _, _ = get_dataloaders(
        Config.PHASE2_RES, Config.PHASE2_BATCH_SIZE
    )

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.PHASE2_LR, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.PHASE2_EPOCHS
    )

    train_phase(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=Config.PHASE2_EPOCHS,
        device=device,
        patience=Config.EARLY_STOPPING_PATIENCE,
        save_name="checkpoint_phase2.pth",
    )

    # ==========================================
    # Phase 3: High Resolution Fine-Tuning
    # ==========================================
    print("\n=== Phase 3: High Resolution Fine-Tuning ===")
    # Loaders for Phase 3 (High Res)
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        Config.PHASE3_RES, Config.PHASE3_BATCH_SIZE
    )

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.PHASE3_LR, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.PHASE3_EPOCHS
    )

    # Note: This phase will overwrite best_model.pth with the best model from this phase
    train_phase(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=Config.PHASE3_EPOCHS,
        device=device,
        patience=Config.EARLY_STOPPING_PATIENCE,
        save_name="checkpoint_phase3.pth",
    )

    # ==========================================
    # Final Validation & Analysis
    # ==========================================
    print("\n=== Final Validation & Analysis ===")

    # Load best model (saved by train_phase)
    best_model_path = os.path.join(Config.OUTPUT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        load_checkpoint(best_model_path, model, device=device)
    else:
        print("Warning: best_model.pth not found. Using current model state.")

    # Validate
    val_loss = validate(model, val_loader, device)
    print(f"Final Validation Metric: {val_loss}")

    # Failure Analysis
    analyze_failures(model, val_loader, Config.VAL_METADATA, device)

    # ==========================================
    # Submission
    # ==========================================
    THRESHOLD = 0.14004325100369866

    if val_loss < THRESHOLD:
        print(
            f"\nValidation loss {val_loss} is below threshold {THRESHOLD}. Generating submission..."
        )
        generate_submission(model, test_loader, classes, device, Config.SUBMISSION_PATH)
    else:
        print(
            f"\nValidation loss {val_loss} is NOT below threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
