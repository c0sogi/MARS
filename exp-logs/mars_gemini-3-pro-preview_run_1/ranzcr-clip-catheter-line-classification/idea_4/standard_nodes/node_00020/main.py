import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from provided library files
from library.config import Config
from library.dataset import get_dataloader
from library.model import MultiTaskEfficientNet
from library.engine import train_one_epoch, valid_one_epoch, inference_fn
from library.utils import seed_everything, get_auc_score, ModelEma


def run_failure_analysis(y_true, y_pred):
    """
    Analyzes the correlation between model error and input features (label count).
    """
    # Calculate Mean Absolute Error per sample
    # y_true and y_pred are (N, 11)
    mae_per_sample = np.mean(np.abs(y_true - y_pred), axis=1)

    # Calculate label count (complexity)
    label_counts = np.sum(y_true, axis=1)

    # Calculate correlation
    if np.std(mae_per_sample) > 0 and np.std(label_counts) > 0:
        correlation = np.corrcoef(mae_per_sample, label_counts)[0, 1]
    else:
        correlation = 0.0

    print(
        f"Failure Analysis - Correlation between Error (MAE) and Label Count: {correlation:.4f}"
    )
    return correlation


def get_validation_preds(model, dataloader, device):
    """
    Helper to get raw predictions and targets for analysis.
    """
    model.eval()
    preds = []
    targets_list = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            targets = batch["targets"].to(device)

            # Forward pass
            cls_logits, _ = model(images)
            batch_preds = torch.sigmoid(cls_logits)

            preds.append(batch_preds.cpu().numpy())
            targets_list.append(targets.cpu().numpy())

    return np.concatenate(preds, axis=0), np.concatenate(targets_list, axis=0)


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Device: {device}")

    # Override Config
    # 12 epochs on full data
    NUM_EPOCHS = 12

    # 2. Data Loading
    print("Loading Data...")
    # Train loader
    train_loader = get_dataloader(
        metadata_path=Config.TRAIN_METADATA,
        mode="train",
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        load_cached_data=True,
    )

    # Val loader
    val_loader = get_dataloader(
        metadata_path=Config.VAL_METADATA,
        mode="val",
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        load_cached_data=True,
    )

    # 3. Model Setup
    print("Initializing Model...")
    model = MultiTaskEfficientNet(pretrained=True)
    model.to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # EMA
    ema_model = None
    if Config.USE_EMA:
        ema_model = ModelEma(model, decay=Config.EMA_DECAY, device=device)

    # 4. Training Loop
    best_auc = 0.0

    print("Starting Training...")
    for epoch in range(NUM_EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            dataloader=train_loader,
            device=device,
            epoch=epoch,
            ema_model=ema_model,
        )

        # Validation (using EMA if available)
        eval_model = ema_model.module if ema_model else model
        val_loss, val_auc = valid_one_epoch(
            model=eval_model, dataloader=val_loader, device=device
        )

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val AUC: {val_auc:.4f}"
        )

        # Save Best
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(eval_model.state_dict(), Config.MODEL_PATH)

    print("Training Complete.")

    # 5. Final Evaluation & Failure Analysis
    print("Loading Best Model for Analysis...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    # Get predictions on validation set
    val_preds, val_targets = get_validation_preds(model, val_loader, device)

    # Compute Final Metric
    final_metric = get_auc_score(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    run_failure_analysis(val_targets, val_preds)

    # 6. Submission
    THRESHOLD = 0.9529163070786033

    if final_metric > THRESHOLD:
        print("Metric exceeds threshold. Generating submission...")

        # Load Test Loader
        test_loader = get_dataloader(
            metadata_path=Config.TEST_METADATA,
            mode="test",
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            load_cached_data=True,
        )

        # Inference
        test_preds = inference_fn(model, test_loader, device)

        # Create Submission DataFrame
        sample_sub = pd.read_csv(Config.TEST_METADATA)

        submission_df = pd.DataFrame(test_preds, columns=Config.TARGET_COLS)
        submission_df.insert(0, "StudyInstanceUID", sample_sub["StudyInstanceUID"])

        # Save
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"Metric {final_metric} did not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
