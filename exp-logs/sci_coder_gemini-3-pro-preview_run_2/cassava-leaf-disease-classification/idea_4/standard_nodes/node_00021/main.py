import os
import sys
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pointbiserialr

# Import from provided library files
from library.config import Config
from library.dataset import get_dataloaders
from library.model import CassavaConvNext
from library.engine import train_one_epoch, evaluate, predict_with_tta
from library.utils import seed_everything, get_logger, SoftTargetCrossEntropy


def get_val_predictions(model, loader, device):
    """
    Runs inference on the validation set to get predictions and labels
    for failure analysis.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


def main():
    # 1. Configuration and Setup
    cfg = Config()

    # Override config for specific task requirements
    cfg.submission_path = "./submission/submission.csv"
    os.makedirs(os.path.dirname(cfg.submission_path), exist_ok=True)

    # Ensure reproducibility
    seed_everything(cfg.seed)

    # Setup Logger
    logger = get_logger(os.path.join(cfg.working_dir, "run.log"))
    logger.info("Starting Cassava Disease Classification Task")

    # 2. Data Loading
    logger.info("Loading DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    # 3. Model Initialization
    logger.info(f"Initializing Model: {cfg.model_name}")
    model = CassavaConvNext(cfg)
    model = model.to(cfg.device)

    # 4. Optimizer, Scheduler, Loss
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr
    )

    # SoftTargetCrossEntropy for MixUp/CutMix during training
    train_criterion = SoftTargetCrossEntropy()
    # Standard CrossEntropy for validation
    val_criterion = nn.CrossEntropyLoss()

    # 5. Training Loop
    best_acc = 0.0
    best_loss = float("inf")

    logger.info(f"Starting training for {cfg.epochs} epochs...")

    for epoch in range(cfg.epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, cfg.device, train_criterion
        )

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, cfg.device, val_criterion)

        # Update Scheduler
        scheduler.step()

        epoch_time = time.time() - start_time
        logger.info(
            f"Epoch {epoch+1}/{cfg.epochs} | Time: {epoch_time:.1f}s | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            best_loss = val_loss
            torch.save(model.state_dict(), cfg.best_model_path)
            logger.info(f"New best model saved with accuracy: {best_acc:.4f}")
        elif val_acc == best_acc and val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), cfg.best_model_path)
            logger.info(f"New best model saved with lower loss: {best_loss:.4f}")

    # 6. Final Validation Assessment
    logger.info("Loading best model for final assessment...")
    if os.path.exists(cfg.best_model_path):
        model.load_state_dict(torch.load(cfg.best_model_path, map_location=cfg.device))

    final_loss, final_acc = evaluate(model, val_loader, cfg.device, val_criterion)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_acc}")

    # 7. Failure Analysis
    logger.info("Performing Failure Analysis...")
    val_preds, val_labels = get_val_predictions(model, val_loader, cfg.device)

    # Calculate errors (1 for error, 0 for correct)
    errors = (val_preds != val_labels).astype(int)

    # Load validation metadata to get file paths
    val_df = pd.read_csv(cfg.val_metadata_path)

    # Calculate file sizes
    file_sizes = []
    for rel_path in val_df["file_path"]:
        full_path = os.path.join(cfg.input_dir, rel_path)
        try:
            file_sizes.append(os.path.getsize(full_path))
        except:
            file_sizes.append(0)

    # Compute correlation
    if len(set(errors)) > 1:  # Correlation requires variance
        corr, p_value = pointbiserialr(errors, file_sizes)
        print(
            f"Correlation between Error and File Size: {corr:.4f} (p-value: {p_value:.4f})"
        )
    else:
        print(
            "Cannot compute correlation: No variance in errors (Model is either 100% correct or 100% wrong)."
        )

    # 8. Conditional Submission
    THRESHOLD = 0.8995994659546062

    if final_acc > THRESHOLD:
        logger.info(
            f"Validation accuracy {final_acc} > {THRESHOLD}. Generating submission..."
        )

        # Generate predictions with TTA
        test_ids, test_preds = predict_with_tta(model, test_loader, cfg.device)

        # Create submission dataframe
        submission_df = pd.DataFrame({"image_id": test_ids, "label": test_preds})

        # Save
        submission_df.to_csv(cfg.submission_path, index=False)
        logger.info(f"Submission saved to {cfg.submission_path}")
    else:
        logger.info(
            f"Validation accuracy {final_acc} <= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
