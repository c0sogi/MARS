import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
import numpy as np
import warnings
from ast import literal_eval

# Import provided library components
from library.utils import seed_everything
from library.dataset import get_dataset
from library.model import ResNet18UNet
from library.engine import train_one_epoch, evaluate, predict_test

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
BATCH_SIZE = 32
EPOCHS = 6
LR = 1e-4
SEED = 42
METADATA_DIR = "./metadata"
OUTPUT_DIR = "./working/idea_11"
SUBMISSION_DIR = "./submission"
THRESHOLD = 0.49944536565378


def perform_failure_analysis(model, loader, device, df_val):
    """
    Analyzes model errors on the validation set and correlates them with metadata.
    """
    model.eval()
    criterion_cls = nn.CrossEntropyLoss(reduction="none")
    criterion_seg = nn.BCEWithLogitsLoss(reduction="none")

    losses = []

    # Calculate per-sample loss
    with torch.no_grad():
        for batch_idx, (images, masks, labels) in enumerate(loader):
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            mask_preds, cls_logits = model(images)

            # Study Loss (N,)
            target_cls = torch.argmax(labels, dim=1)
            loss_cls = criterion_cls(cls_logits, target_cls)

            # Segmentation Loss (N, 1, H, W) -> mean over spatial dims -> (N,)
            loss_seg_map = criterion_seg(mask_preds, masks)
            loss_seg = loss_seg_map.mean(dim=(1, 2, 3))

            # Composite Loss
            batch_loss = loss_cls + 10.0 * loss_seg
            losses.extend(batch_loss.cpu().numpy())

    # Add losses to dataframe
    # Ensure alignment: loader is shuffle=False, df_val is same order
    analysis_df = df_val.copy()
    # Truncate if sizes mismatch (e.g. drop_last in loader, though val usually doesn't drop)
    analysis_df = analysis_df.iloc[: len(losses)]
    analysis_df["error_magnitude"] = losses

    # Extract features for correlation
    # 1. Class Label (0-3)
    def get_class_idx(row):
        if row["Negative for Pneumonia"]:
            return 0
        if row["Typical Appearance"]:
            return 1
        if row["Indeterminate Appearance"]:
            return 2
        if row["Atypical Appearance"]:
            return 3
        return 0

    analysis_df["class_idx"] = analysis_df.apply(get_class_idx, axis=1)

    # 2. Number of Boxes
    def get_num_boxes(x):
        try:
            if pd.isna(x) or x == "":
                return 0
            return len(literal_eval(x))
        except:
            return 0

    analysis_df["num_boxes"] = analysis_df["boxes"].apply(get_num_boxes)

    # Calculate correlations
    correlations = analysis_df[["error_magnitude", "class_idx", "num_boxes"]].corr()[
        "error_magnitude"
    ]

    print("\n==== Failure Analysis ====")
    print("Correlation between Error Magnitude and Input Features:")
    print(correlations.drop("error_magnitude"))
    print("==========================\n")


def main():
    # 1. Setup
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # 2. Load Metadata
    df_val = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    df_test = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # 3. Data Loading
    # Using load_cached_data=True to use preprocessed .npy files if available
    train_ds = get_dataset("train", load_cached_data=True)
    val_ds = get_dataset("val", load_cached_data=True)
    test_ds = get_dataset("test", load_cached_data=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    # 4. Model Initialization
    model = ResNet18UNet(num_classes=4, pretrained=True)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)

    # Scheduler
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS * steps_per_epoch
    )

    scaler = GradScaler()

    # 5. Training Loop
    best_map = 0.0
    best_model_path = os.path.join(OUTPUT_DIR, "best_model.pth")

    print(f"Starting training on {device} for {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device
        )
        val_loss, val_map = evaluate(model, val_loader, device, df_val)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{EPOCHS} | Time: {elapsed:.1f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mAP: {val_map:.6f}"
        )

        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), best_model_path)

    # 6. Final Evaluation & Metrics
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(best_model_path))

    # Re-evaluate to get the exact metric on the full set (already done in loop, but ensuring consistency)
    _, final_map = evaluate(model, val_loader, device, df_val)

    print(f"Final Validation Metric: {final_map}")

    # 7. Failure Analysis
    perform_failure_analysis(model, val_loader, device, df_val)

    # 8. Submission
    if final_map > THRESHOLD:
        print(
            f"Validation metric ({final_map}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        predict_test(model, test_loader, device, df_test)
    else:
        print(
            f"Validation metric ({final_map}) did not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
