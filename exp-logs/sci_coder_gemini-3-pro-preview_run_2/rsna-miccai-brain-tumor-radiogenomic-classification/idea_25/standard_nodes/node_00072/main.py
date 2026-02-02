import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Import provided library modules
from library import config
from library import utils
from library import dataset
from library import model as lib_model
from library import train_eval
from library import inference


def main():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration
    # -------------------------------------------------------------------------
    # Ensure reproducibility
    utils.seed_everything(config.SEED)

    # Setup logger
    logger = utils.get_logger("runfile")
    device = config.DEVICE
    logger.info(f"Running on device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    logger.info("Initializing datasets...")

    # Training Dataset
    train_ds = dataset.RSNADataset(
        split="train", transform=dataset.get_transforms("train"), debug=config.DEBUG
    )

    # Validation Dataset
    val_ds = dataset.RSNADataset(
        split="val", transform=dataset.get_transforms("valid"), debug=config.DEBUG
    )

    logger.info(f"Training samples: {len(train_ds)}")
    logger.info(f"Validation samples: {len(val_ds)}")

    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(device == "cuda"),
    )

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    logger.info("Initializing model...")
    net = lib_model.AsymmetricEfficientNet()
    net = net.to(device)

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    logger.info("Starting training...")

    # Run training (handles optimization, logging, and checkpointing)
    # Returns the model with the best weights loaded
    net = train_eval.run_training(
        model=net,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.NUM_EPOCHS,
        device=device,
        patience=config.EARLY_STOPPING_PATIENCE,
    )

    # -------------------------------------------------------------------------
    # 5. Final Validation Assessment
    # -------------------------------------------------------------------------
    logger.info("Performing final validation assessment...")

    # Ensure model is in eval mode
    net.eval()
    criterion = nn.BCEWithLogitsLoss()

    # Calculate metric on the full validation set
    val_loss, val_auc = train_eval.validate(net, val_loader, criterion, device, logger)

    # REQUIRED OUTPUT: Print the final validation metric
    print(f"Final Validation Metric: {val_auc}")

    # -------------------------------------------------------------------------
    # 6. Failure Analysis
    # -------------------------------------------------------------------------
    logger.info("Running failure analysis on validation set...")

    all_targets = []
    all_probs = []

    # Collect predictions for failure analysis
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, dtype=torch.float32)
            logits = net(images)
            probs = torch.sigmoid(logits)

            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy().flatten())

    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Calculate Error Magnitude
    errors = np.abs(all_targets - all_probs)

    # Prepare DataFrame for correlation analysis
    # val_ds.df aligns with the loader because shuffle=False
    analysis_df = val_ds.df.copy()
    analysis_df["error"] = errors
    analysis_df["target"] = all_targets

    # Extract a simple structural feature: Number of FLAIR slices
    # This acts as a proxy for brain volume / scan resolution
    flair_counts = []
    for idx, row in analysis_df.iterrows():
        try:
            full_path = os.path.join(config.INPUT_DIR, row["path_FLAIR"])
            # Quick count of files
            count = len([f for f in os.listdir(full_path) if f.endswith(".dcm")])
        except Exception:
            count = 0
        flair_counts.append(count)

    analysis_df["flair_slice_count"] = flair_counts

    # Calculate Correlations
    corr_slice = analysis_df["flair_slice_count"].corr(analysis_df["error"])
    corr_target = analysis_df["target"].corr(analysis_df["error"])

    print("Failure Analysis - Correlation with Error Magnitude:")
    print(f"Correlation (FLAIR Slice Count vs Error): {corr_slice}")
    print(f"Correlation (Target Class vs Error): {corr_target}")

    # -------------------------------------------------------------------------
    # 7. Submission Generation
    # -------------------------------------------------------------------------
    # Threshold defined in the task
    THRESHOLD = 0.6303636363636363

    if val_auc > THRESHOLD:
        logger.info(
            f"Validation AUC ({val_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        # Define output path
        submission_file = "submission.csv"

        # Run inference pipeline
        inference.generate_submission(
            model_weights_path=os.path.join(config.WORKING_DIR, "best_model.pth"),
            output_file=submission_file,
            device=device,
            debug=config.DEBUG,
        )
    else:
        logger.warning(
            f"Validation AUC ({val_auc}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
