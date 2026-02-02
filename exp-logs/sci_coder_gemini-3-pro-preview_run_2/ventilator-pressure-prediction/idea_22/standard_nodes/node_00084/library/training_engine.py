import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import seed_everything, MetricMonitor
from library.model_architecture import PADIBiLSTM, WeightedL1Loss
from library.data_factory import get_dataloaders


class Trainer:
    """
    Manages the training and validation process, including metric tracking,
    checkpointing, and early stopping.
    """

    def __init__(
        self,
        model,
        device,
        optimizer,
        scheduler,
        criterion,
        patience=20,
        checkpoint_path=None,
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.patience = patience
        self.checkpoint_path = checkpoint_path or os.path.join(
            Config.WORKING_DIR, "best_model.pth"
        )

        self.best_mae = float("inf")
        self.counter = 0
        self.early_stop = False

    def train_epoch(self, train_loader):
        self.model.train()
        monitor = MetricMonitor()

        for batch in train_loader:
            X = batch["X"].to(self.device)
            y = batch["y"].to(self.device)
            u_out = batch["u_out"].to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(X)
            loss = self.criterion(pred, y, u_out)
            loss.backward()
            self.optimizer.step()

            monitor.update("Loss", loss.item(), X.size(0))

        return monitor.get_avg("Loss")

    def validate(self, val_loader):
        self.model.eval()
        monitor = MetricMonitor()

        with torch.no_grad():
            for batch in val_loader:
                X = batch["X"].to(self.device)
                y = batch["y"].to(self.device)
                u_out = batch["u_out"].to(self.device)

                pred = self.model(X)
                loss = self.criterion(pred, y, u_out)

                # Metric: MAE on Inspiratory Phase (u_out == 0)
                mask = u_out == 0
                if mask.sum() > 0:
                    mae = torch.abs(pred[mask] - y[mask]).mean()
                    monitor.update("MAE_Insp", mae.item(), mask.sum().item())

                monitor.update("Loss", loss.item(), X.size(0))

        return monitor.get_avg("Loss"), monitor.get_avg("MAE_Insp")

    def fit(self, train_loader, val_loader, epochs):
        print(f"Starting training for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            self.scheduler.step()
            val_loss, val_mae = self.validate(val_loader)

            # Print full precision metrics as requested
            print(
                f"Epoch {epoch}: Train Loss: {train_loss} | Val Loss: {val_loss} | Val MAE Insp: {val_mae}"
            )

            # Checkpointing & Early Stopping
            if val_mae < self.best_mae:
                self.best_mae = val_mae
                self.counter = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(f"New best model saved with MAE: {self.best_mae}")
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    self.early_stop = True
                    break


def run_training(debug=Config.DEBUG, epochs=Config.EPOCHS):
    """
    Main function to setup and run the training pipeline.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # 1. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(debug=debug)

    # 2. Model Initialization
    model = PADIBiLSTM().to(device)

    # 3. Optimizer & Scheduler
    # Stretched-Horizon Protocol: CosineAnnealing with T_max = Total Epochs
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=Config.ETA_MIN)

    # 4. Loss Function
    criterion = WeightedL1Loss()

    # 5. Training
    trainer = Trainer(
        model=model,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        patience=30,  # Allow sufficient time for convergence
        checkpoint_path=os.path.join(Config.WORKING_DIR, "best_model.pth"),
    )

    trainer.fit(train_loader, val_loader, epochs)

    # 6. Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(trainer.checkpoint_path))
    model.eval()

    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            X = batch["X"].to(device)
            pred = model(X)
            predictions.append(pred.cpu().numpy().flatten())

    all_preds = np.concatenate(predictions)

    # 7. Submission Generation
    test_meta = pd.read_csv(Config.TEST_METADATA)

    # Handle potential length mismatch in debug mode
    if len(all_preds) != len(test_meta):
        print(
            f"Warning: Prediction length {len(all_preds)} does not match Metadata length {len(test_meta)}"
        )
        if debug:
            test_meta = test_meta.iloc[: len(all_preds)]

    test_meta["pressure"] = all_preds

    # Format: id, pressure
    submission = test_meta[["id", "pressure"]]
    submission.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
