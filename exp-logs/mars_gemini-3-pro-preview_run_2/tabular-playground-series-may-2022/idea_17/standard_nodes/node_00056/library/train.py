import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from library.config import Config
from library.utils import seed_everything, get_device, compute_auc
from library.data import get_dataloaders
from library.model import HybridTransformerResFunnel


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0.0

    for batch in loader:
        x_seq = batch["x_seq"].to(device)
        x_raw = batch["x_raw"].to(device)
        x_binned = batch["x_binned"].to(device)
        y = batch["target"].to(device).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(x_seq, x_raw, x_binned)
        loss = criterion(logits, y)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    """
    Evaluates the model on a given loader. Returns AUC and predictions.
    """
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            x_seq = batch["x_seq"].to(device)
            x_raw = batch["x_raw"].to(device)
            x_binned = batch["x_binned"].to(device)

            logits = model(x_seq, x_raw, x_binned)
            probs = torch.sigmoid(logits).squeeze(1)

            preds.append(probs.cpu().numpy())

            # Collect targets if they exist (validation set)
            if "target" in batch:
                targets.append(batch["target"].numpy())

    preds = np.concatenate(preds)

    auc = None
    if len(targets) > 0:
        targets = np.concatenate(targets)
        auc = compute_auc(targets, preds)

    return auc, preds


class Trainer:
    """
    Manages the training lifecycle, including optimization, logging, and checkpointing.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        device,
        config,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.config = config
        self.best_auc = 0.0

    def fit(self, epochs, patience):
        patience_counter = 0

        print(f"Starting training on {self.device}...")

        for epoch in range(epochs):
            # Train
            avg_train_loss = train_one_epoch(
                self.model,
                self.train_loader,
                self.optimizer,
                self.criterion,
                self.device,
            )

            # Validate
            val_auc, _ = evaluate(self.model, self.val_loader, self.device)

            # Scheduler Step
            if self.scheduler:
                self.scheduler.step()

            # Logging (Full precision)
            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss} | Val AUC: {val_auc}"
            )

            # Checkpointing & Early Stopping
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                patience_counter = 0
                torch.save(self.model.state_dict(), self.config.MODEL_SAVE_PATH)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        print(f"Training complete. Best Val AUC: {self.best_auc}")


def train(
    debug_subset=None, epochs=Config.EPOCHS, patience=Config.EARLY_STOPPING_PATIENCE
):
    """
    Main training function. Sets up data, model, and trainer.
    """
    seed_everything(Config.RANDOM_STATE)
    device = get_device()

    # Load Data
    train_loader, val_loader, _ = get_dataloaders(
        load_cached_data=True, debug_subset=debug_subset
    )

    # Initialize Model
    model = HybridTransformerResFunnel().to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=Config.SCHEDULER_STEP_SIZE, gamma=Config.SCHEDULER_GAMMA
    )

    # Loss
    criterion = nn.BCEWithLogitsLoss()

    # Trainer
    trainer = Trainer(
        model, train_loader, val_loader, optimizer, scheduler, criterion, device, Config
    )

    trainer.fit(epochs, patience)


def inference():
    """
    Generates submission using the best saved model.
    """
    device = get_device()
    _, _, test_loader = get_dataloaders(load_cached_data=True)

    model = HybridTransformerResFunnel().to(device)

    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Loading model from {Config.MODEL_SAVE_PATH}")
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        print("Warning: No model checkpoint found. Using untrained model.")

    print("Generating predictions...")
    _, preds = evaluate(model, test_loader, device)

    # Load sample submission
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
    sample_sub["target"] = preds

    # Save submission
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    sample_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
