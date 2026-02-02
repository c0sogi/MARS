import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm

from library.config import Config
from library.utils import (
    seed_everything,
    AverageMeter,
    save_checkpoint,
    load_checkpoint,
    laplace_log_likelihood_metric,
)
from library.data import get_dataloaders
from library.model import DynamicDepthGeMNet


class CustomLaplaceLoss(nn.Module):
    """
    Directly optimizes the competition metric:
    Loss = (sqrt(2) * Delta) / sigma_clipped + ln(sqrt(2) * sigma_clipped)
    where Delta is absolute error clipped at 1000, and sigma_clipped is sigma clipped at 70.
    """

    def __init__(self):
        super(CustomLaplaceLoss, self).__init__()

    def forward(self, fvc_pred, sigma_pred, fvc_true):
        # fvc_true might be (B, 1), ensure shapes match
        if fvc_true.dim() == 1:
            fvc_true = fvc_true.view(-1, 1)

        # 1. Clip sigma at 70 ml
        sigma_clipped = torch.clamp(sigma_pred, min=Config.CONFIDENCE_CLIP)

        # 2. Calculate absolute error and clip at 1000 ml
        delta = torch.abs(fvc_true - fvc_pred)
        delta = torch.clamp(delta, max=Config.MAX_ERROR)

        # 3. Compute Loss (Negative of the metric)
        sqrt_2 = torch.sqrt(torch.tensor(2.0, device=fvc_pred.device))

        # Metric term 1: - (sqrt(2) * delta) / sigma
        # Metric term 2: - ln(sqrt(2) * sigma)
        # Loss = -Metric
        loss = (sqrt_2 * delta) / sigma_clipped + torch.log(sqrt_2 * sigma_clipped)

        return torch.mean(loss)


def train_one_epoch(loader, model, criterion, optimizer, device, epoch):
    """
    Handles one epoch of training.
    """
    model.train()
    losses = AverageMeter()

    # Iterate over batches (tqdm removed for silent execution if needed,
    # but kept simple loop as per instructions to not print progress bars)
    for batch_idx, (inputs, target) in enumerate(loader):
        # Move inputs to device
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        target = target.to(device)

        # Forward pass
        fvc_pred, sigma_pred = model(inputs)

        # Calculate loss
        loss = criterion(fvc_pred, sigma_pred, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        losses.update(loss.item(), inputs["img_ax"].size(0))

    return losses.avg


def valid_one_epoch(loader, model, device):
    """
    Handles validation. Returns the competition metric score.
    """
    model.eval()
    scores = AverageMeter()

    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(loader):
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            target = target.to(device)

            fvc_pred, sigma_pred = model(inputs)

            # Calculate metric using the utility function
            metric = laplace_log_likelihood_metric(target, fvc_pred, sigma_pred)
            scores.update(metric, inputs["img_ax"].size(0))

    return scores.avg


def generate_submission(loader, model, test_df, device):
    """
    Generates predictions for the test set and saves the submission file.
    """
    model.eval()
    fvc_preds = []
    sigma_preds = []

    print("Generating predictions for test set...")

    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(loader):
            for k, v in inputs.items():
                inputs[k] = v.to(device)

            fvc_pred, sigma_pred = model(inputs)

            fvc_preds.append(fvc_pred.cpu().numpy())
            sigma_preds.append(sigma_pred.cpu().numpy())

    # Concatenate all batches
    fvc_preds = np.concatenate(fvc_preds, axis=0).flatten()
    sigma_preds = np.concatenate(sigma_preds, axis=0).flatten()

    # The test loader is sequential and matches test_df order
    submission = test_df[["Patient_Week"]].copy()
    submission["FVC"] = fvc_preds
    submission["Confidence"] = sigma_preds

    # Save
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission.head())


def run_training():
    """
    Main execution function.
    """
    seed_everything(Config.SEED)

    # 1. Load Metadata
    print("Loading metadata...")
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # 2. Get Dataloaders
    print("Preparing dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders(train_df, val_df, test_df)

    # 3. Initialize Model, Loss, Optimizer
    print(f"Initializing model on {Config.DEVICE}...")
    model = DynamicDepthGeMNet()
    model = model.to(Config.DEVICE)

    criterion = CustomLaplaceLoss()

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # 4. Training Loop
    best_score = -float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(
            train_loader, model, criterion, optimizer, Config.DEVICE, epoch
        )

        # Validation
        val_score = valid_one_epoch(val_loader, model, Config.DEVICE)

        # Step Scheduler
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Score: {val_score}"
        )

        # Checkpoint & Early Stopping
        if val_score > best_score:
            print(f"Score Improved ({best_score} -> {val_score}). Saving model...")
            best_score = val_score
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_score": best_score,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                is_best=True,
                filename=os.path.join(Config.CACHE_DIR, f"checkpoint_ep{epoch+1}.pth"),
                best_filename=Config.MODEL_SAVE_PATH,
            )
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # 5. Inference
    print("Loading best model for inference...")
    # Re-initialize model to ensure clean state or load weights directly
    model = DynamicDepthGeMNet().to(Config.DEVICE)
    load_checkpoint(Config.MODEL_SAVE_PATH, model, device=Config.DEVICE)

    generate_submission(test_loader, model, test_df, Config.DEVICE)
