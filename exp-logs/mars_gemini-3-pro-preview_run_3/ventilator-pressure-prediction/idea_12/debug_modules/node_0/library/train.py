import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from library.config import Config
from library.utils import seed_everything, AverageMeter, get_device, compute_metric
from library.data import prepare_datasets
from library.model import LANNet


class MaskedL1Loss(nn.Module):
    """
    Computes L1 Loss (MAE) only for the inspiratory phase (u_out == 0).
    """

    def __init__(self):
        super().__init__()

    def forward(self, preds, targets, u_out):
        # Create mask: True where u_out is 0 (inspiratory phase)
        mask = u_out == 0

        # Select valid elements
        preds_masked = preds[mask]
        targets_masked = targets[mask]

        # Compute MAE
        if preds_masked.numel() == 0:
            return torch.tensor(0.0, device=preds.device, requires_grad=True)

        loss = torch.abs(preds_masked - targets_masked).mean()
        return loss


class Trainer:
    def __init__(self, model, optimizer, scheduler, device, criterion, patience=15):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.criterion = criterion
        self.patience = patience
        self.best_val_mae = float("inf")
        self.counter = 0
        self.early_stop = False

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        losses = AverageMeter()

        # Disable progress bar for cleaner logs in non-interactive environments,
        # or keep it simple. We'll iterate silently or with minimal print to avoid log clutter.
        for batch_idx, (x, y, u_out) in enumerate(train_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            u_out = u_out.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            preds = self.model(x)

            # Compute loss
            loss = self.criterion(preds, y, u_out)

            # Backward pass
            loss.backward()

            # Gradient clipping (optional but recommended for LSTMs)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            losses.update(loss.item(), x.size(0))

        return losses.avg

    def validate(self, val_loader):
        self.model.eval()
        losses = AverageMeter()

        with torch.no_grad():
            for x, y, u_out in val_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                u_out = u_out.to(self.device)

                preds = self.model(x)
                loss = self.criterion(preds, y, u_out)

                losses.update(loss.item(), x.size(0))

        return losses.avg

    def fit(self, train_loader, val_loader, epochs):
        print(f"Starting training for {epochs} epochs on {self.device}...")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader)

            # Scheduler step
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]["lr"]
            else:
                current_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train MAE: {train_loss:.6f} | "
                f"Val MAE: {val_loss} | "  # Printing full precision as requested
                f"LR: {current_lr:.2e}"
            )

            # Checkpointing and Early Stopping
            if val_loss < self.best_val_mae:
                self.best_val_mae = val_loss
                self.counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_CHECKPOINT)
                print(f"  -> Model saved! New best Val MAE: {self.best_val_mae}")
            else:
                self.counter += 1
                print(f"  -> No improvement. Counter: {self.counter}/{self.patience}")

            # Save last model
            torch.save(self.model.state_dict(), Config.LAST_MODEL_CHECKPOINT)

            if self.counter >= self.patience:
                print("Early stopping triggered.")
                self.early_stop = True
                break


def generate_submission(model, test_loader, device):
    """
    Generates predictions for the test set and saves the submission file.
    """
    print("Generating submission...")
    model.eval()
    predictions = []

    with torch.no_grad():
        for x, _, _ in test_loader:
            x = x.to(device)
            preds = model(x)
            predictions.append(preds.cpu().numpy().flatten())

    # Concatenate all predictions
    predictions = np.concatenate(predictions)

    # Load Test IDs from cache
    # Note: prepare_datasets saves test_ids.npy in the working directory
    if not os.path.exists(Config.TEST_CACHE_IDS):
        raise FileNotFoundError(
            f"Test IDs not found at {Config.TEST_CACHE_IDS}. Run data prep first."
        )

    test_ids = np.load(Config.TEST_CACHE_IDS).flatten()

    # Ensure lengths match
    if len(test_ids) != len(predictions):
        raise ValueError(
            f"Length mismatch: IDs ({len(test_ids)}) vs Preds ({len(predictions)})"
        )

    # Create DataFrame
    submission = pd.DataFrame({Config.ID_COL: test_ids, Config.TARGET_COL: predictions})

    # Save
    submission.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")


def train_model(debug=False, epochs=Config.EPOCHS):
    """
    Main function to execute the training pipeline.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()

    # 2. Data Loading
    train_dataset, val_dataset, test_dataset = prepare_datasets(
        load_cached_data=True, debug=debug
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Helps with batch norm stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = LANNet(config=Config).to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    criterion = MaskedL1Loss()

    # 5. Training
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        criterion=criterion,
        patience=15,  # Early stopping patience
    )

    trainer.fit(train_loader, val_loader, epochs)

    # 6. Inference
    # Load best model weights
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT, map_location=device))

    generate_submission(model, test_loader, device)
