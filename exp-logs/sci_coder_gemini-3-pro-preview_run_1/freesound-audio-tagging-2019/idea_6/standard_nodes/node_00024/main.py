import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import soundfile as sf
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import set_seed, calculate_lwlrap
from library.dataset import AudioDataset, collate_fn
from library.model import AudioClassifier
from library.engine import fit, validate


def run_pipeline():
    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # Ensure output directory exists
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # 2. Data Loading
    print("Initializing Datasets...")
    train_dataset = AudioDataset(mode="train")
    val_dataset = AudioDataset(mode="val")

    # Use num_workers from Config
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # Validation loader uses collate_fn for variable length audio
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = AudioClassifier(
        num_classes=Config.NUM_CLASSES,
        backbone=Config.BACKBONE,
        pretrained=Config.PRETRAINED,
    )
    model = model.to(device)

    # 4. Optimizer and Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    # 5. Training
    print("Starting Training...")
    # fit() handles the training loop, validation per epoch, and saving the best model
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=Config.EPOCHS,
    )

    # 6. Final Validation & Metric Calculation
    print("Loading best model for final validation...")
    best_model_path = os.path.join(Config.OUTPUT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: Best model not found. Using current model weights.")

    criterion = nn.BCEWithLogitsLoss()
    val_loss, val_lwlrap = validate(model, val_loader, criterion, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_lwlrap}")

    # 7. Failure Analysis
    print("Performing Failure Analysis...")
    model.eval()

    val_bce_losses = []
    val_durations = []
    val_label_counts = []

    # We need to compute per-sample loss and gather metadata
    # Re-loading validation metadata to get filepaths for duration calculation
    val_meta_df = pd.read_csv(Config.VAL_CSV)

    # Pre-calculate durations to align with loader order
    # The loader iterates sequentially because shuffle=False
    # However, let's be robust and map by fname if possible, but the loader order matches the CSV order.

    # Calculate durations
    print("Calculating validation audio durations...")
    durations_map = {}
    for idx, row in val_meta_df.iterrows():
        path = os.path.join(Config.INPUT_ROOT, row["filepath"])
        try:
            info = sf.info(path)
            durations_map[row["fname"]] = info.duration
        except:
            durations_map[row["fname"]] = 0.0

    # Collect predictions and targets
    all_preds = []
    all_targets = []
    all_fnames = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            fnames = batch["fname"]

            logits = model(images)
            # BCE Loss per sample (reduce='none' then mean over classes)
            bce_loss = nn.functional.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )
            bce_loss = bce_loss.mean(dim=1).cpu().numpy()

            val_bce_losses.extend(bce_loss)
            all_fnames.extend(fnames)

            # For label count
            targets_np = targets.cpu().numpy()
            counts = targets_np.sum(axis=1)
            val_label_counts.extend(counts)

    # Align durations
    val_durations = [durations_map.get(f, 0.0) for f in all_fnames]

    # Compute Correlations
    loss_arr = np.array(val_bce_losses)
    dur_arr = np.array(val_durations)
    count_arr = np.array(val_label_counts)

    corr_duration = np.corrcoef(loss_arr, dur_arr)[0, 1]
    corr_count = np.corrcoef(loss_arr, count_arr)[0, 1]

    print(f"Correlation between Error (BCE Loss) and Audio Duration: {corr_duration}")
    print(f"Correlation between Error (BCE Loss) and Label Count: {corr_count}")

    # 8. Submission
    THRESHOLD = 0.8554930465617762

    if val_lwlrap > THRESHOLD:
        print(
            f"Validation metric {val_lwlrap} exceeds threshold {THRESHOLD}. Generating submission..."
        )

        test_dataset = AudioDataset(mode="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        test_preds = []
        test_fnames = []

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(device)
                fnames = batch["fname"]

                logits = model(images)
                preds = torch.sigmoid(logits)

                test_preds.append(preds.cpu().numpy())
                test_fnames.extend(fnames)

        test_preds = np.concatenate(test_preds, axis=0)

        # Create Submission DataFrame
        # Columns must match sample_submission.csv
        # The AudioDataset.label_cols are derived from test.csv which is derived from sample_submission
        # So the order is preserved.

        sub_df = pd.DataFrame(test_preds, columns=test_dataset.label_cols)
        sub_df.insert(0, "fname", test_fnames)

        submission_path = "./submission/submission.csv"
        os.makedirs("./submission", exist_ok=True)
        sub_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(
            f"Validation metric {val_lwlrap} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    run_pipeline()
