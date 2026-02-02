import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import get_device, log_message, print_metrics
from library.model import BirdResNetSPP


class Trainer:
    """
    Trainer class for managing the training, validation, and prediction processes
    for the Bird Species Classification model.
    """

    def __init__(self, model, device=None):
        """
        Args:
            model (nn.Module): The PyTorch model to train.
            device (torch.device, optional): Device to run training on. Defaults to auto-detect.
        """
        self.device = device if device else get_device()
        self.model = model.to(self.device)

        # Loss function: Binary Cross Entropy with Logits for Multi-label classification
        self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer: AdamW
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler: Cosine Annealing
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
        )

    def mixup_data(self, x, y, alpha=Config.MIXUP_ALPHA):
        """
        Applies Mixup augmentation to inputs and targets.
        Returns:
            mixed_x: Mixed input images.
            y_a: Targets for the first image set.
            y_b: Targets for the second image set.
            lam: Lambda mixing coefficient.
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(self.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, pred, y_a, y_b, lam):
        """
        Calculates loss for mixed predictions.
        """
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)

    def train_one_epoch(self, train_loader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        for images, labels, _ in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            if Config.USE_MIXUP:
                images, targets_a, targets_b, lam = self.mixup_data(images, labels)
                outputs = self.model(images)
                loss = self.mixup_criterion(outputs, targets_a, targets_b, lam)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            loss.backward()

            # Gradient Clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

            self.optimizer.step()

            running_loss += loss.item() * images.size(0)

        self.scheduler.step()

        epoch_loss = running_loss / len(train_loader.dataset)
        return epoch_loss

    def validate(self, val_loader):
        """
        Runs validation on the validation set.
        Returns:
            val_loss (float): Average validation loss.
            val_auc (float): Macro-averaged ROC AUC score.
        """
        self.model.eval()
        running_loss = 0.0
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)

                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(outputs)
                all_targets.append(labels.cpu().numpy())
                all_preds.append(probs.cpu().numpy())

        val_loss = running_loss / len(val_loader.dataset)

        all_targets = np.vstack(all_targets)
        all_preds = np.vstack(all_preds)

        # Calculate ROC AUC robustly
        # Scikit-learn's roc_auc_score throws an error if a class has only one label (all 0 or all 1)
        # We calculate AUC per class and average, skipping problematic classes for the validation metric
        aucs = []
        for i in range(all_targets.shape[1]):
            try:
                # Check if class exists in targets (has at least one 0 and one 1)
                if len(np.unique(all_targets[:, i])) > 1:
                    score = roc_auc_score(all_targets[:, i], all_preds[:, i])
                    aucs.append(score)
            except ValueError:
                pass

        val_auc = np.mean(aucs) if aucs else 0.5

        return val_loss, val_auc

    def fit(self, train_loader, val_loader, fold_idx):
        """
        Main training loop with Early Stopping.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            fold_idx: Index of the current fold (for saving checkpoints).

        Returns:
            best_auc (float): The best validation AUC achieved.
        """
        best_auc = 0.0
        patience_counter = 0
        best_model_path = f"{Config.CHECKPOINT_DIR}/fold_{fold_idx}_best.pth"

        log_message(f"Starting training for Fold {fold_idx}...")

        for epoch in range(Config.EPOCHS):
            train_loss = self.train_one_epoch(train_loader)
            val_loss, val_auc = self.validate(val_loader)

            metrics = {
                "Epoch": epoch + 1,
                "Train Loss": train_loss,
                "Val Loss": val_loss,
                "Val AUC": val_auc,
            }
            # Print metrics with full precision
            print_metrics(metrics, prefix=f"Fold {fold_idx}")

            # Checkpoint and Early Stopping Logic
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
            else:
                patience_counter += 1

            if patience_counter >= Config.PATIENCE:
                log_message(
                    f"Early stopping triggered at epoch {epoch + 1} for Fold {fold_idx}"
                )
                break

        log_message(f"Training finished for Fold {fold_idx}. Best AUC: {best_auc}")

        # Load best model weights before returning
        self.model.load_state_dict(
            torch.load(best_model_path, map_location=self.device)
        )
        return best_auc

    def predict(self, test_loader):
        """
        Generates predictions for the test set.

        Args:
            test_loader: DataLoader for test data.

        Returns:
            ids (list): List of recording IDs.
            probs (np.ndarray): Predicted probabilities (N_samples, N_classes).
        """
        self.model.eval()
        all_ids = []
        all_probs = []

        with torch.no_grad():
            for images, _, rec_ids in test_loader:
                images = images.to(self.device)

                outputs = self.model(images)
                probs = torch.sigmoid(outputs)

                all_probs.append(probs.cpu().numpy())
                all_ids.extend(rec_ids.numpy())

        return all_ids, np.vstack(all_probs)
