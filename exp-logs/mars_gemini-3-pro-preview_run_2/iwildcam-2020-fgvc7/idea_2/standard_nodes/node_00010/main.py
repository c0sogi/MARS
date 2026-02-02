import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset

from library.config import Config
from library.utils import seed_everything, get_device
from library.data import get_dataloaders, MixupCutmixCollator
from library.model import get_model
from library.engine import train_one_epoch, validate
from library.inference import generate_submission


def main():
    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    # Override Config for fast baseline execution
    Config.NUM_EPOCHS = 4
    Config.BATCH_SIZE = 16
    TRAIN_SAMPLE_LIMIT = 40000

    seed_everything(Config.SEED)
    device = get_device()

    print("Initializing pipeline...")
    print(f"Device: {device}")
    print(f"Epochs: {Config.NUM_EPOCHS}")
    print(f"Batch Size: {Config.BATCH_SIZE}")

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    # Load all data first (sample_size=None to ensure full val/test sets)
    train_loader_full, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, sample_size=None
    )

    # Manually subset the training dataset to limit runtime
    train_dataset_full = train_loader_full.dataset
    if len(train_dataset_full) > TRAIN_SAMPLE_LIMIT:
        print(f"Subsetting training data to {TRAIN_SAMPLE_LIMIT} samples...")
        # Use fixed indices for reproducibility
        indices = range(TRAIN_SAMPLE_LIMIT)
        train_dataset = Subset(train_dataset_full, indices)
    else:
        train_dataset = train_dataset_full

    # Re-create Train Loader with Mixup Collator
    mixup_collator = MixupCutmixCollator(
        num_classes=Config.NUM_CLASSES,
        mixup_alpha=Config.MIXUP_ALPHA,
        cutmix_alpha=Config.CUTMIX_ALPHA,
        prob=Config.MIXUP_PROB,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=mixup_collator,
        pin_memory=True,
        drop_last=True,
    )

    # --------------------------------------------------------------------------
    # 3. Model & Optimizer
    # --------------------------------------------------------------------------
    model = get_model(pretrained=True)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS, eta_min=Config.MIN_LR
    )

    # AMP Scaler
    scaler = torch.cuda.amp.GradScaler()

    # --------------------------------------------------------------------------
    # 4. Training Loop
    # --------------------------------------------------------------------------
    print("Starting training...")
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        train_stats = train_one_epoch(
            model, train_loader, optimizer, device, scaler, epoch
        )
        scheduler.step()

    # --------------------------------------------------------------------------
    # 5. Final Validation
    # --------------------------------------------------------------------------
    print("Running final validation...")
    val_metrics = validate(model, val_loader, device)
    final_acc = val_metrics["accuracy"]

    # REQUIRED FORMAT
    print(f"Final Validation Metric: {final_acc}")

    # --------------------------------------------------------------------------
    # 6. Failure Analysis
    # --------------------------------------------------------------------------
    print("Performing Failure Analysis...")
    model.eval()

    all_preds = []
    all_targets = []
    all_ids = []

    # Collect predictions on validation set
    with torch.no_grad():
        for images, targets, img_ids in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_ids.extend(img_ids)

    # Create DataFrame
    results_df = pd.DataFrame(
        {"id": all_ids, "prediction": all_preds, "target": all_targets}
    )

    # Calculate Error Magnitude (1 if incorrect, 0 if correct)
    results_df["error_magnitude"] = (
        results_df["prediction"] != results_df["target"]
    ).astype(int)

    # Load Metadata
    meta_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Merge
    analysis_df = pd.merge(results_df, meta_df, on="id", how="left")

    # Calculate Correlations
    if "width" in analysis_df.columns and "height" in analysis_df.columns:
        corr_width = analysis_df["error_magnitude"].corr(analysis_df["width"])
        corr_height = analysis_df["error_magnitude"].corr(analysis_df["height"])

        print(f"Correlation between Error and Width: {corr_width}")
        print(f"Correlation between Error and Height: {corr_height}")
    else:
        print("Metadata missing width/height columns, skipping correlation analysis.")

    # --------------------------------------------------------------------------
    # 7. Submission
    # --------------------------------------------------------------------------
    THRESHOLD = 0.7304232880255179

    if final_acc > THRESHOLD:
        print(
            f"Validation accuracy {final_acc} > {THRESHOLD}. Generating submission..."
        )
        generate_submission(model, test_loader, device)
    else:
        print(f"Validation accuracy {final_acc} <= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
