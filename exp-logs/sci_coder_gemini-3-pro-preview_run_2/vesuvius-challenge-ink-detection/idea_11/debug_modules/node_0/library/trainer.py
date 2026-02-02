import os
import torch
import torch.nn as nn
from library.config import Config
from library.utils import seed_everything, fbeta_score
from library.losses import DiceBCELoss


class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, device):
        """
        Initializes the Trainer.

        Args:
            model: The PyTorch model to train.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            optimizer: The optimizer.
            scheduler: The learning rate scheduler.
            device: The device to run training on.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.criterion = DiceBCELoss()

        # Initialize best score with baseline.
        # We only save if we exceed the provided baseline score.
        self.best_score = Config.BASELINE_SCORE

        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        self.save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

        # Early Stopping parameters
        self.early_stopping_patience = 5
        self.epochs_no_improve = 0

        # Set seeds for reproducibility
        seed_everything(Config.SEED)

    def train_epoch(self, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        for batch_idx, (images, labels, masks, _) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Accumulate loss (weighted by batch size for accurate mean)
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        return epoch_loss

    def validate(self):
        """
        Runs validation on the validation set.
        Returns average loss and F0.5 score.
        """
        self.model.eval()
        running_loss = 0.0
        dataset_size = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, labels, masks, _ in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                batch_size = images.size(0)
                running_loss += loss.item() * batch_size
                dataset_size += batch_size

                # Apply sigmoid to convert logits to probabilities
                probs = torch.sigmoid(outputs)

                # Store predictions and targets on CPU for global metric calculation
                all_preds.append(probs.cpu())
                all_targets.append(labels.cpu())

        val_loss = running_loss / dataset_size

        # Concatenate all batches
        all_preds = torch.cat(all_preds, dim=0).view(-1)
        all_targets = torch.cat(all_targets, dim=0).view(-1)

        # Compute F0.5 Score
        val_f05 = fbeta_score(
            all_preds, all_targets, beta=Config.F_BETA, threshold=Config.MASK_THRESHOLD
        )

        return val_loss, val_f05

    def fit(self, num_epochs=Config.NUM_EPOCHS):
        """
        Main training loop.
        """
        print(f"Starting training on device: {self.device}")

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, val_f05 = self.validate()

            # Step the scheduler
            # ReduceLROnPlateau expects a metric. We use validation loss.
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            # Print metrics with full precision
            print(
                f"Epoch {epoch}: Train Loss: {train_loss}, Val Loss: {val_loss}, Val F0.5: {val_f05}"
            )

            # Checkpoint Logic
            if val_f05 > self.best_score:
                print(
                    f"Validation score improved from {self.best_score} to {val_f05}. Saving model to {self.save_path}"
                )
                self.best_score = val_f05
                torch.save(self.model.state_dict(), self.save_path)
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
                print(
                    f"No improvement. Best: {self.best_score}. Patience: {self.epochs_no_improve}/{self.early_stopping_patience}"
                )

            # Early Stopping
            if self.epochs_no_improve >= self.early_stopping_patience:
                print("Early stopping triggered.")
                break
