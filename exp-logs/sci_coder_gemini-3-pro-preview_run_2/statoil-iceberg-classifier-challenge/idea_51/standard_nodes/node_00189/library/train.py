import copy
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import log_loss
from library import config, utils, model, data


class Trainer:
    """
    Manages the training and validation process for a single model instance.
    """

    def __init__(
        self, model, train_loader, val_loader, criterion, optimizer, scheduler, device
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logger = utils.get_logger("trainer")

    def train_one_epoch(self, epoch_index):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        for batch_idx, (images, angles, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            angles = angles.to(self.device)
            labels = labels.to(self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass
            # Model expects (x_img, x_inc)
            outputs = self.model(images, angles)

            # Ensure labels are (N, 1) to match outputs
            loss = self.criterion(outputs, labels.view(-1, 1))

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            # Statistics
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        return epoch_loss

    def validate(self):
        """
        Runs validation on the validation set.
        Returns average loss and log_loss metric.
        """
        self.model.eval()
        running_loss = 0.0
        dataset_size = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, angles, labels in self.val_loader:
                images = images.to(self.device)
                angles = angles.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images, angles)
                loss = self.criterion(outputs, labels.view(-1, 1))

                batch_size = images.size(0)
                running_loss += loss.item() * batch_size
                dataset_size += batch_size

                # Collect for sklearn log_loss
                all_preds.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        epoch_loss = running_loss / dataset_size

        # Concatenate all batches
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # Calculate Log Loss
        # Clip predictions to avoid log(0) errors, though BCELoss handles this internally,
        # sklearn's log_loss is robust but explicit clipping is safe practice for float32.
        # However, we'll pass raw probabilities as sklearn handles it.
        metric_log_loss = log_loss(all_labels, all_preds, labels=[0, 1])

        return epoch_loss, metric_log_loss

    def fit(self, num_epochs, patience):
        """
        Executes the training loop with Early Stopping and Learning Rate Scheduling.
        """
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(num_epochs):
            train_loss = self.train_one_epoch(epoch)
            val_loss, val_metric = self.validate()

            # Update Learning Rate
            if self.scheduler:
                self.scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]["lr"]
            else:
                current_lr = config.LEARNING_RATE

            # Logging (Full Precision)
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"LR: {current_lr} - "
                f"Train Loss: {train_loss} - "
                f"Val Loss: {val_loss} - "
                f"Val LogLoss: {val_metric}"
            )

            # Early Stopping Logic
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
                # self.logger.info("Validation loss improved. Saving model.")
            else:
                patience_counter += 1
                # self.logger.info(f"EarlyStopping counter: {patience_counter} out of {patience}")
                if patience_counter >= patience:
                    self.logger.info("Early stopping triggered.")
                    break

        # Load best model weights
        self.model.load_state_dict(best_model_wts)
        return best_model_wts


def run_fold(fold_idx, data_dict, scaler):
    """
    Sets up and trains the model for a specific fold.

    Args:
        fold_idx (int): The fold index.
        data_dict (dict): The processed data dictionary.
        scaler (GlobalScaler): The fitted scaler.

    Returns:
        dict: The state dictionary of the best trained model.
    """
    logger = utils.get_logger(f"fold_{fold_idx}")
    logger.info(f"Starting training for Fold {fold_idx}")

    # 1. Get DataLoaders
    train_loader, val_loader, _ = data.get_dataloaders(fold_idx, data_dict, scaler)

    # 2. Initialize Model
    # RDP_WBN is defined in library.model
    net = model.RDP_WBN()
    net.to(config.DEVICE)

    # 3. Initialize Optimizer
    # Using Adam as per strategy
    optimizer = torch.optim.Adam(net.parameters(), lr=config.LEARNING_RATE)

    # 4. Initialize Scheduler
    # ReduceLROnPlateau as per strategy
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.SCHEDULER_FACTOR,
        patience=config.SCHEDULER_PATIENCE,
        min_lr=config.SCHEDULER_MIN_LR,
    )

    # 5. Initialize Criterion
    # Binary Cross Entropy Loss
    criterion = nn.BCELoss()

    # 6. Initialize Trainer
    trainer = Trainer(
        model=net,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config.DEVICE,
    )

    # 7. Run Training
    best_weights = trainer.fit(num_epochs=config.NUM_EPOCHS, patience=config.PATIENCE)

    logger.info(f"Fold {fold_idx} completed.")
    return best_weights
