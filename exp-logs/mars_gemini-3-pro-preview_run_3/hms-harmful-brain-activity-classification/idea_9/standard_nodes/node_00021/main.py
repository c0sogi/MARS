import sys
import os
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
import logging

# Ensure library can be imported
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import get_dataloaders, EEGDataset
from library.model import get_model
from library.engine import train_one_epoch, validate, inference
from library.transforms import MixUp, get_transforms
from torch.utils.data import DataLoader


def main():
    # ==========================================
    # 1. Setup & Configuration
    # ==========================================
    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    # Initialize Logger
    logger = get_logger(os.path.join(Config.OUTPUT_DIR, "run.log"))
    logger.info("Starting Fast Baseline Pipeline...")

    # Device
    device = torch.device(Config.DEVICE)

    # Override Config for Fast Baseline Execution
    # We use a subset of data and fewer epochs to ensure completion within strict time limits
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20000  # Train on ~25% of data for speed
    Config.EPOCHS = 5  # Sufficient for baseline convergence
    Config.BATCH_SIZE = 32

    logger.info(f"Device: {device}")
    logger.info(f"Training Samples: {Config.DEBUG_SAMPLE_SIZE}")
    logger.info(f"Epochs: {Config.EPOCHS}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    logger.info("Initializing DataLoaders...")
    # Get training loader (subsampled via debug flag) and val loader (subsampled)
    # We will manually load the FULL validation set later for the final metric
    dataloaders = get_dataloaders(load_cached_data=True, debug=Config.DEBUG)
    train_loader = dataloaders["train"]
    val_loader_sub = dataloaders["val"]  # Used for monitoring during training

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    logger.info("Initializing Siamese Equivariant Network...")
    model = get_model(pretrained=True)
    model.to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    # Augmentation
    mixup_fn = MixUp(alpha=0.5, prob=0.5)

    # ==========================================
    # 4. Training Loop
    # ==========================================
    logger.info("Starting Training...")
    best_loss = float("inf")

    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch, mixup_fn
        )

        # Validate (on subsample for speed)
        val_loss = validate(model, val_loader_sub, device)

        logger.info(
            f"Epoch {epoch} - Train Loss: {train_loss:.4f} - Val Loss (Sub): {val_loss:.4f}"
        )

        # Scheduler Step
        scheduler.step()

        # Checkpoint
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            logger.info(f"  Saved Best Model (Loss: {best_loss:.4f})")

    # ==========================================
    # 5. Final Evaluation (Full Validation Set)
    # ==========================================
    logger.info("\nLoading Best Model for Final Evaluation...")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    logger.info("Loading Full Validation Set...")
    # Manually create dataset/loader for full validation to ensure metric accuracy
    val_df_full = pd.read_csv(Config.VAL_CSV)
    val_ds_full = EEGDataset(
        val_df_full, mode="val", transforms=get_transforms(mode="val")
    )
    val_loader_full = DataLoader(
        val_ds_full,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Compute Metric and Collect Errors
    criterion = torch.nn.KLDivLoss(reduction="none")
    total_loss_sum = 0.0
    total_count = 0
    all_losses = []

    logger.info("Computing Final Metrics...")
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(val_loader_full):
            # Move to device
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.to(device)
            targets = targets.to(device)

            # Forward
            logits = model(data["eeg"], data["spec"])
            log_probs = F.log_softmax(logits, dim=1)

            # KL Divergence per sample: Sum over classes
            # shape: (Batch, Classes) -> sum(1) -> (Batch,)
            loss_per_sample = criterion(log_probs, targets).sum(dim=1)

            # Accumulate
            total_loss_sum += loss_per_sample.sum().item()
            total_count += targets.size(0)

            # Store for analysis
            all_losses.extend(loss_per_sample.cpu().numpy())

    final_metric = total_loss_sum / total_count

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # 6. Failure Analysis
    # ==========================================
    logger.info("\nPerforming Failure Analysis...")

    # Attach errors to dataframe
    # Ensure lengths match (loader might drop last if configured, but here drop_last=False)
    if len(all_losses) == len(val_df_full):
        val_df_full["error"] = all_losses

        # Correlation Analysis
        meta_cols = [
            "eeg_label_offset_seconds",
            "spectogram_label_offset_seconds",
            "total_votes",
        ]

        print("Correlation between Error (KL Div) and Metadata:")
        for col in meta_cols:
            if col in val_df_full.columns:
                corr = val_df_full["error"].corr(val_df_full[col])
                print(f"  {col}: {corr:.6f}")
    else:
        logger.warning(
            f"Mismatch in validation sizes: DF {len(val_df_full)} vs Preds {len(all_losses)}"
        )

    # ==========================================
    # 7. Submission
    # ==========================================
    THRESHOLD = 0.8169508603799445

    if final_metric < THRESHOLD:
        logger.info(
            f"\nMetric ({final_metric:.6f}) meets threshold ({THRESHOLD:.6f}). Generating Submission..."
        )

        # Check for test data
        if os.path.exists(Config.TEST_CSV):
            test_df = pd.read_csv(Config.TEST_CSV)
            if not test_df.empty:
                # Create Test Loader
                test_ds = EEGDataset(
                    test_df, mode="test", transforms=get_transforms(mode="test")
                )
                test_loader = DataLoader(
                    test_ds,
                    batch_size=Config.BATCH_SIZE,
                    shuffle=False,
                    num_workers=Config.NUM_WORKERS,
                    pin_memory=True,
                )

                # Run Inference
                inference(model, test_loader, device, Config.SUBMISSION_PATH)
            else:
                logger.warning("Test CSV is empty.")
        else:
            logger.warning("Test CSV not found.")
    else:
        logger.info(
            f"\nMetric ({final_metric:.6f}) did not meet threshold ({THRESHOLD:.6f}). Submission skipped."
        )


if __name__ == "__main__":
    main()
