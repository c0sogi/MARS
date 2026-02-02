import os
import numpy as np
import torch
from library.utils import calculate_roc_auc


class Trainer:
    """
    Trainer class encapsulating the training, validation, and inference logic
    for the Right Whale Detection model.
    """

    def __init__(self, model, criterion, optimizer, scheduler, device, config):
        """
        Initialize the Trainer.

        Args:
            model (nn.Module): The neural network model.
            criterion (nn.Module): The loss function.
            optimizer (Optimizer): The optimizer.
            scheduler (LRScheduler): The learning rate scheduler.
            device (torch.device): The device to run on (CPU/GPU).
            config (class): Configuration class containing hyperparameters.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    def train_epoch(self, train_loader):
        """
        Runs one epoch of training with Mixup augmentation.

        Args:
            train_loader (DataLoader): DataLoader for training data.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(self.device)
            # Ensure targets are (B, 1) for BCEWithLogitsLoss
            targets = targets.to(self.device).unsqueeze(1)
            batch_size = inputs.size(0)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
                # Apply Mixup Augmentation
                if self.config.MIXUP_ALPHA > 0:
                    lam = np.random.beta(
                        self.config.MIXUP_ALPHA, self.config.MIXUP_ALPHA
                    )
                    index = torch.randperm(batch_size).to(self.device)

                    mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
                    targets_a, targets_b = targets, targets[index]

                    outputs = self.model(mixed_inputs)

                    # Compute loss by mixing the weighted scalar losses of the input pairs
                    loss = lam * self.criterion(outputs, targets_a) + (
                        1 - lam
                    ) * self.criterion(outputs, targets_b)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
        return epoch_loss

    def validate(self, val_loader):
        """
        Evaluates the model on the validation set.

        Args:
            val_loader (DataLoader): DataLoader for validation data.

        Returns:
            tuple: (Average Validation Loss, Validation AUC)
        """
        self.model.eval()
        running_loss = 0.0
        dataset_size = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device).unsqueeze(1)
                batch_size = inputs.size(0)

                with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                running_loss += loss.item() * batch_size
                dataset_size += batch_size

                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(outputs)
                all_preds.append(probs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

        if len(all_preds) > 0:
            all_preds = np.concatenate(all_preds)
            all_targets = np.concatenate(all_targets)
            auc = calculate_roc_auc(all_targets, all_preds)
        else:
            auc = 0.5

        return epoch_loss, auc

    def fit(self, train_loader, val_loader, save_path, epochs):
        """
        Runs the full training loop with early stopping.

        Args:
            train_loader (DataLoader): Training data.
            val_loader (DataLoader): Validation data.
            save_path (str): Path to save the best model checkpoint.
            epochs (int): Maximum number of epochs.

        Returns:
            float: Best validation AUC achieved.
        """
        best_auc = 0.0
        patience_counter = 0

        # Ensure the directory for the checkpoint exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_auc = self.validate(val_loader)

            if self.scheduler:
                self.scheduler.step()

            # Print metrics with full precision
            print(
                f"Epoch {epoch+1} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
            )

            # Checkpoint and Early Stopping Logic
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(self.model.state_dict(), save_path)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.PATIENCE:
                print("Early stopping triggered.")
                break

        # Load the best model weights before returning
        if os.path.exists(save_path):
            self.model.load_state_dict(torch.load(save_path, map_location=self.device))

        return best_auc

    def predict(self, loader):
        """
        Generates predictions for a dataset.

        Args:
            loader (DataLoader): DataLoader for inference data.

        Returns:
            np.ndarray: Flattened array of predicted probabilities.
        """
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(self.device)
                with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
                    outputs = self.model(inputs)
                probs = torch.sigmoid(outputs)
                all_preds.append(probs.cpu().numpy())

        if len(all_preds) > 0:
            return np.concatenate(all_preds).flatten()
        else:
            return np.array([])
