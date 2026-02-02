import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from transformers import get_linear_schedule_with_warmup

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, get_logger, get_device
from library.data import get_dataloaders
from library.model import SiameseDeberta
from library.engine import train_one_epoch, validate, predict


def main():
    # 1. Setup
    Config.setup()
    seed_everything(Config.seed)
    logger = get_logger("runfile")

    # Override Config for Fast Baseline
    # We use 1 epoch to ensure completion within 2 hours.
    Config.epochs = 1
    Config.debug = False

    logger.info("Configuration configured for fast baseline.")
    logger.info(f"Epochs: {Config.epochs}")
    logger.info(f"Debug Mode: {Config.debug}")

    # 2. Data Loading
    logger.info("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    device = get_device()
    logger.info(f"Initializing model on {device}...")
    model = SiameseDeberta()
    model.to(device)

    # 4. Optimizer and Scheduler Setup
    # Replicating logic from engine.py since we need to pass these to train_one_epoch
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": Config.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=Config.learning_rate)

    # Calculate steps
    num_update_steps_per_epoch = len(train_loader) // Config.gradient_accumulation_steps
    max_train_steps = Config.epochs * num_update_steps_per_epoch

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * max_train_steps),
        num_training_steps=max_train_steps,
    )

    scaler = GradScaler(enabled=Config.fp16)

    # 5. Training Loop
    best_loss = float("inf")
    logger.info("Starting training loop...")

    for epoch in range(Config.epochs):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, scaler, epoch
        )

        # Validate
        val_metrics = validate(model, val_loader, device)
        val_loss = val_metrics["log_loss"]  # Use log_loss as the primary metric

        logger.info(
            f"Epoch {epoch+1}/{Config.epochs} | Train Loss: {train_loss:.6f} | Val Log Loss: {val_loss:.6f}"
        )

        # Save Best
        if val_loss < best_loss:
            best_loss = val_loss
            logger.info(f"Validation improved. Saving model to {Config.model_path}")
            torch.save(model.state_dict(), Config.model_path)
        else:
            logger.info("Validation did not improve.")

    # 6. Final Evaluation & Failure Analysis
    logger.info("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(Config.model_path, map_location=device))

    # Compute Final Metric
    final_metrics = validate(model, val_loader, device)
    final_log_loss = final_metrics["log_loss"]

    # REQUIRED PRINT FORMAT
    print(f"Final Validation Metric: {final_log_loss:.16f}")

    # Failure Analysis
    logger.info("Performing failure analysis...")

    # Collect targets and scalars from validation set
    # Note: val_loader shuffle is False, so order is preserved
    all_targets = []
    all_scalars = []

    # Iterate loader to get ground truth and features
    # We don't need gradients here
    for batch in val_loader:
        # batch['target'] is (B, 3)
        all_targets.append(batch["target"].numpy())
        # batch['scalars'] is (B, 3) -> [P_len, Ra_len, Rb_len] (log scale)
        all_scalars.append(batch["scalars"].numpy())

    all_targets = np.concatenate(all_targets, axis=0)
    all_scalars = np.concatenate(all_scalars, axis=0)

    # Get predictions
    val_preds = predict(model, val_loader, device)

    # Calculate Error Magnitude (Log Loss per sample)
    # Clip predictions to prevent log(0)
    epsilon = 1e-15
    val_preds_clipped = np.clip(val_preds, epsilon, 1 - epsilon)

    # Cross Entropy per sample: - sum(target * log(pred))
    sample_losses = -np.sum(all_targets * np.log(val_preds_clipped), axis=1)

    # Calculate Correlations
    # Scalar 0: Prompt Log Len
    # Scalar 1: Response A Log Len
    # Scalar 2: Response B Log Len
    corr_prompt = np.corrcoef(sample_losses, all_scalars[:, 0])[0, 1]
    corr_res_a = np.corrcoef(sample_losses, all_scalars[:, 1])[0, 1]
    corr_res_b = np.corrcoef(sample_losses, all_scalars[:, 2])[0, 1]

    print("Failure Analysis - Correlation with Error Magnitude:")
    print(f"Prompt Length: {corr_prompt:.4f}")
    print(f"Response A Length: {corr_res_a:.4f}")
    print(f"Response B Length: {corr_res_b:.4f}")

    # 7. Conditional Submission
    threshold = 1.0005665522536111

    if final_log_loss < threshold:
        logger.info(
            f"Validation metric {final_log_loss} < {threshold}. Generating submission..."
        )

        # Generate predictions on test set
        test_preds = predict(model, test_loader, device)

        # Load Test IDs
        test_df = pd.read_csv(Config.test_path)

        # Ensure lengths match (handling potential debug truncation)
        if len(test_preds) != len(test_df):
            logger.warning(
                "Mismatch in prediction and test ID count. Truncating IDs to match predictions."
            )
            test_df = test_df.iloc[: len(test_preds)]

        # Create Submission DataFrame
        submission = pd.DataFrame(
            {
                "id": test_df["id"],
                "winner_model_a": test_preds[:, 0],
                "winner_model_b": test_preds[:, 1],
                "winner_tie": test_preds[:, 2],
            }
        )

        submission.to_csv(Config.submission_path, index=False)
        logger.info(f"Submission saved to {Config.submission_path}")
    else:
        logger.info(
            f"Validation metric {final_log_loss} >= {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
