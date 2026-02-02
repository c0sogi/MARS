import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

# Import from provided library files
from library.config import Config
from library.utils import set_seed, get_logger
from library.dataset import get_dataloaders
from library.model import get_model
from library.trainer import Trainer


def main():
    # 1. Configuration
    # Set epochs to 2 to ensure the run completes within the 2-hour limit while using the full dataset.
    cfg = Config(epochs=2)

    # 2. Reproducibility
    set_seed(cfg.seed)

    # 3. Logger
    logger = get_logger(cfg.log_path)
    logger.info("Starting execution of runfile.py")

    # 4. Data Loading
    logger.info("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    # 5. Model Setup
    logger.info(f"Initializing model: {cfg.model_name}")
    model = get_model(cfg)

    # 6. Optimizer and Scheduler
    # AdamW is standard for Transformers
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=1e-6
    )

    # 7. Training
    trainer = Trainer(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    logger.info("Starting training...")
    trainer.fit()

    # 8. Final Validation & Failure Analysis
    logger.info("Performing final validation and failure analysis...")

    # Load best model weights
    if os.path.exists(cfg.best_model_path):
        checkpoint = torch.load(cfg.best_model_path, map_location=cfg.device)
        # Handle state dict keys if needed (Trainer saves standard state_dict)
        state_dict = checkpoint["state_dict"]
        model.load_state_dict(state_dict)
        logger.info(
            f"Loaded best model from epoch {checkpoint.get('epoch', 'unknown')}"
        )
    else:
        logger.warning("Best model not found. Using current model weights.")

    model.eval()

    all_preds = []
    all_targets = []

    # Run inference on validation set
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(cfg.device)
            # Use mixed precision for inference speed
            with torch.cuda.amp.autocast():
                outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.numpy())

    # Calculate Final Metric
    final_f1 = f1_score(all_targets, all_preds, average="macro")
    print(f"Final Validation Metric: {final_f1}")

    # Failure Analysis
    # Load validation metadata to correlate errors with features
    val_df = pd.read_csv(cfg.val_csv_path)

    # Handle debug mode subsetting if applicable
    if cfg.debug:
        val_df = val_df.head(cfg.debug_sample_size)

    # Ensure alignment
    if len(val_df) == len(all_preds):
        val_df["predicted"] = all_preds
        val_df["target"] = all_targets
        val_df["error"] = (val_df["predicted"] != val_df["target"]).astype(int)

        # Calculate correlation with Region ID
        if "region_id" in val_df.columns:
            corr_region = val_df["error"].corr(val_df["region_id"])
            print(f"Correlation between Error and Region ID: {corr_region}")
        else:
            logger.warning("region_id column not found in validation metadata.")
    else:
        logger.warning(
            f"Mismatch in validation set size: DataFrame {len(val_df)} vs Preds {len(all_preds)}"
        )

    # 9. Submission
    threshold = 0.43008749389564027
    if final_f1 > threshold:
        logger.info(
            f"Validation metric {final_f1} exceeds threshold {threshold}. Generating submission..."
        )
        trainer.predict()
    else:
        logger.info(
            f"Validation metric {final_f1} does not exceed threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
