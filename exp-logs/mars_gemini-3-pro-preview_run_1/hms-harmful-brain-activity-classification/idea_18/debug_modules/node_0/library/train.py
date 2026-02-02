import os
import torch
import pandas as pd
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from library.config import Config
from library.utils import seed_everything, KLDivLossWithLogits, AverageMeter
from library.data import get_dataloader
from library.models import BottleneckProjectedFusionNet


def train_one_epoch(loader, model, optimizer, scheduler, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    losses = AverageMeter()

    for batch_idx, (eeg, spec, target) in enumerate(loader):
        eeg = eeg.to(device, non_blocking=True)
        spec = spec.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward pass
        logits = model(eeg, spec)

        # Compute loss
        loss = criterion(logits, target)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Optimizer and Scheduler steps
        optimizer.step()
        scheduler.step()

        losses.update(loss.item(), eeg.size(0))

    return losses.avg


def validate_one_epoch(loader, model, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    losses = AverageMeter()

    with torch.no_grad():
        for eeg, spec, target in loader:
            eeg = eeg.to(device, non_blocking=True)
            spec = spec.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            logits = model(eeg, spec)
            loss = criterion(logits, target)

            losses.update(loss.item(), eeg.size(0))

    return losses.avg


def train_model():
    """
    Main function to orchestrate the training process.
    """
    seed_everything(Config.SEED)

    # Ensure output directory exists
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    print(f"Loading metadata...")
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)

    print("Initializing DataLoaders...")
    train_loader = get_dataloader(
        train_df, Config, mode="train", batch_size=Config.BATCH_SIZE, shuffle=True
    )
    val_loader = get_dataloader(
        val_df, Config, mode="val", batch_size=Config.BATCH_SIZE, shuffle=False
    )

    print("Initializing Model...")
    model = BottleneckProjectedFusionNet(Config)
    model = model.to(Config.DEVICE)

    # Optimizer
    optimizer = AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    # Loss Function (KL Divergence)
    criterion = KLDivLossWithLogits()

    # Scheduler (OneCycleLR)
    total_steps = len(train_loader) * Config.EPOCHS
    scheduler = OneCycleLR(
        optimizer,
        max_lr=Config.LR,
        total_steps=total_steps,
        pct_start=0.1,
        div_factor=25,
        final_div_factor=100,
    )

    best_val_loss = float("inf")
    best_model_path = os.path.join(Config.OUTPUT_DIR, "best_model.pth")

    print(f"Starting training for {Config.EPOCHS} epochs on device: {Config.DEVICE}")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            train_loader, model, optimizer, scheduler, criterion, Config.DEVICE
        )
        val_loss = validate_one_epoch(val_loader, model, criterion, Config.DEVICE)

        print(
            f"Epoch {epoch + 1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        # Save best model (Early Stopping mechanism)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path}")

    print(f"Training complete. Best Validation Loss: {best_val_loss:.6f}")


def inference():
    """
    Loads the best model and generates predictions for the test set.
    """
    seed_everything(Config.SEED)

    print(f"Loading test metadata...")
    test_df = pd.read_csv(Config.TEST_CSV)

    print("Initializing Test DataLoader...")
    test_loader = get_dataloader(
        test_df, Config, mode="test", batch_size=Config.BATCH_SIZE, shuffle=False
    )

    print("Loading Best Model...")
    model = BottleneckProjectedFusionNet(Config)
    best_model_path = os.path.join(Config.OUTPUT_DIR, "best_model.pth")

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(
            f"Best model not found at {best_model_path}. Run train_model() first."
        )

    model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))
    model = model.to(Config.DEVICE)
    model.eval()

    print("Generating predictions...")
    all_probs = []

    with torch.no_grad():
        for eeg, spec in test_loader:
            eeg = eeg.to(Config.DEVICE, non_blocking=True)
            spec = spec.to(Config.DEVICE, non_blocking=True)

            logits = model(eeg, spec)

            # Apply Softmax to convert logits to probabilities (sum to 1)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)

    # Format Submission
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    submission_path = os.path.join(submission_dir, "submission.csv")

    sub_df = pd.DataFrame(all_probs, columns=Config.CLASS_NAMES)

    # Insert eeg_id as the first column
    sub_df.insert(0, "eeg_id", test_df["eeg_id"])

    sub_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
