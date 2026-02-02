import os
import time
import copy
import torch
import torch.nn as nn
import numpy as np
from library.utils import seed_everything, compute_multilabel_auc
from library.sam import SAM


class Trainer:
    """
    Trainer class to manage the training process for a single fold.
    Integrates SAM optimizer, Mixup regularization, and Aggressive Early Stopping.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        fold,
        epochs=50,
        patience=10,
        mixup_alpha=1.0,
        checkpoint_dir="./working/idea_21/checkpoints",
    ):
        """
        Args:
            model (nn.Module): The PyTorch model to train.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            optimizer (SAM): The SAM optimizer instance.
            scheduler (lr_scheduler): Learning rate scheduler.
            device (torch.device): Device to run training on.
            fold (int): Current fold number.
            epochs (int): Maximum number of epochs.
            patience (int): Patience for early stopping.
            mixup_alpha (float): Alpha parameter for Beta distribution in Mixup.
            checkpoint_dir (str): Directory to save model checkpoints.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.fold = fold
        self.epochs = epochs
        self.patience = patience
        self.mixup_alpha = mixup_alpha
        self.checkpoint_dir = checkpoint_dir

        # Use BCEWithLogitsLoss for multi-label classification
        self.criterion = nn.BCEWithLogitsLoss()

        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.best_auc = 0.0
        self.best_model_state = None

    def mixup_data(self, x, y):
        """
        Applies Mixup to inputs and targets.
        Returns mixed inputs, mixed targets, and lambda.
        """
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(self.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        mixed_y = lam * y + (1 - lam) * y[index, :]

        return mixed_x, mixed_y

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Apply Mixup
            mixed_images, mixed_labels = self.mixup_data(images, labels)

            # --- SAM Step 1: Compute gradients at current weights ---
            # Forward pass
            outputs = self.model(mixed_images)
            loss = self.criterion(outputs, mixed_labels)

            # Backward pass to populate .grad
            loss.backward()

            # --- SAM Step 2: Perturb weights and recompute gradients ---
            # Define closure for SAM
            def closure():
                # Note: SAM.first_step(zero_grad=True) is called inside optimizer.step()
                # before this closure is executed.
                # We must re-compute the forward pass and loss on the perturbed weights.
                output_closure = self.model(mixed_images)
                loss_closure = self.criterion(output_closure, mixed_labels)
                loss_closure.backward()
                return loss_closure

            # Update weights using SAM
            self.optimizer.step(closure)

            # Zero gradients for next iteration
            self.optimizer.zero_grad()

            running_loss += loss.item()

        epoch_loss = running_loss / len(self.train_loader)
        return epoch_loss

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()

                # Apply sigmoid for probabilities
                preds = torch.sigmoid(outputs)

                all_preds.append(preds.cpu().numpy())
                all_targets.append(labels.cpu().numpy())

        val_loss = running_loss / len(self.val_loader)

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Compute AUC using the robust library function
        val_auc = compute_multilabel_auc(all_targets, all_preds)

        return val_loss, val_auc

    def fit(self):
        print(f"Starting training for Fold {self.fold}...")
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            start_time = time.time()

            train_loss = self.train_one_epoch(epoch)
            val_loss, val_auc = self.validate()

            # Step the scheduler
            if self.scheduler:
                self.scheduler.step()

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch}/{self.epochs} - "
                f"Train Loss: {train_loss:.6f} - "
                f"Val Loss: {val_loss:.6f} - "
                f"Val AUC: {val_auc:.10f} - "  # Full precision
                f"Time: {elapsed:.2f}s"
            )

            # Checkpointing and Early Stopping
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0

                # Save best model
                save_path = os.path.join(
                    self.checkpoint_dir, f"best_model_fold_{self.fold}.pth"
                )
                torch.save(self.best_model_state, save_path)
                # print(f"  New best model saved to {save_path}")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(
                        f"Early stopping triggered at epoch {epoch}. Best Val AUC: {self.best_auc:.10f}"
                    )
                    break

        # Load best weights before returning
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        print(f"Fold {self.fold} finished. Best Val AUC: {self.best_auc:.10f}")
        return self.best_auc

    def predict(self, test_loader):
        """
        Generates predictions for the test set.
        """
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for images in test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                preds = torch.sigmoid(outputs)
                all_preds.append(preds.cpu().numpy())

        return np.concatenate(all_preds, axis=0)
