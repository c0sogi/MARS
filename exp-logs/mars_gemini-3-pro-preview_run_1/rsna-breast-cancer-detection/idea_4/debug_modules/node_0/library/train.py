import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import from the provided library files
from library.config import Config, set_seed
from library.utils import get_device, probabilistic_f1
from library.data import get_dataloaders
from library.model import EarlyFusionEfficientNet


class Trainer:
    """
    Trainer class to manage the training and validation lifecycle of the model.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        config,
        patience=3,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.patience = patience

        self.best_score = -float("inf")
        self.best_model_path = os.path.join(self.config.CACHE_DIR, "best_model.pth")

        # Ensure cache directory exists
        os.makedirs(self.config.CACHE_DIR, exist_ok=True)

    def train_epoch(self, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(self.train_loader):
            inputs = batch["image"].to(self.device)
            # Metadata channels are handled inside the dataset/model pipeline via
            # Spatial Channel Expansion and Stochastic Modality Dropout.
            # We just need to pass the input tensor.

            targets = batch["label"].to(self.device).view(-1, 1)

            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(inputs)
            loss = self.criterion(logits, targets)

            # Backward pass
            loss.backward()

            # Note: Gradient clipping is explicitly disabled as per strategy

            self.optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(self.train_loader)
        return epoch_loss

    def validate(self):
        """
        Runs validation on the validation set.
        """
        self.model.eval()
        running_loss = 0.0
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch["image"].to(self.device)
                targets = batch["label"].to(self.device).view(-1, 1)

                logits = self.model(inputs)
                loss = self.criterion(logits, targets)

                running_loss += loss.item()

                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(logits)

                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(probs.cpu().numpy())

        val_loss = running_loss / len(self.val_loader)

        # Flatten lists
        all_targets = np.concatenate(all_targets).ravel()
        all_preds = np.concatenate(all_preds).ravel()

        # Calculate pF1 score
        pf1 = probabilistic_f1(all_targets, all_preds)

        return val_loss, pf1

    def fit(self, num_epochs):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training on device: {self.device}")
        epochs_no_improve = 0

        for epoch in range(1, num_epochs + 1):
            start_time = time.time()

            train_loss = self.train_epoch(epoch)
            val_loss, val_pf1 = self.validate()

            # Step the scheduler
            if self.scheduler:
                self.scheduler.step()

            end_time = time.time()
            epoch_mins = (end_time - start_time) / 60

            print(f"Epoch {epoch}/{num_epochs} | Time: {epoch_mins} min")
            print(f"Train Loss: {train_loss}")
            print(f"Val Loss: {val_loss}")
            print(f"Val pF1: {val_pf1}")

            # Check for improvement
            if val_pf1 > self.best_score:
                print(
                    f"Validation pF1 improved from {self.best_score} to {val_pf1}. Saving model..."
                )
                self.best_score = val_pf1
                torch.save(self.model.state_dict(), self.best_model_path)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"No improvement in pF1 for {epochs_no_improve} epochs.")

            # Early Stopping
            if epochs_no_improve >= self.patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

        print(f"Training complete. Best Validation pF1: {self.best_score}")
        return self.best_score


def run_training(
    debug=Config.DEBUG, num_epochs=Config.NUM_EPOCHS, batch_size=Config.BATCH_SIZE
):
    """
    Initializes components and runs the training process.
    """
    # 1. Set Seed for Reproducibility
    set_seed(Config.SEED)

    # 2. Setup Device
    device = get_device()

    # 3. Get DataLoaders
    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size, debug=debug)

    # 4. Initialize Model
    model = EarlyFusionEfficientNet(
        model_name=Config.MODEL_NAME,
        pretrained=True,
        dropout_prob=Config.MODALITY_DROPOUT_PROB,
        in_chans=Config.INPUT_CHANNELS,
        num_classes=Config.NUM_CLASSES,
    )
    model = model.to(device)

    # 5. Define Loss Function
    # Using BCEWithLogitsLoss with pos_weight as per strategy
    pos_weight = torch.tensor([Config.POS_WEIGHT], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 6. Define Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # 7. Define Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # 8. Initialize Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=Config,
        patience=3,  # Early stopping patience
    )

    # 9. Start Training
    best_score = trainer.fit(num_epochs=num_epochs)

    return trainer.model, best_score
