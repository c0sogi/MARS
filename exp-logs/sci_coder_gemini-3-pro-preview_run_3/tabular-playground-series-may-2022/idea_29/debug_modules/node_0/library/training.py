import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from library.config import MAX_LR, WEIGHT_DECAY, EPOCHS, WORKING_DIR
from library.utils import save_checkpoint


class Trainer:
    """
    Trainer class to manage the training loop, validation, and optimization
    for the HC-PFE model.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        max_lr: float = MAX_LR,
        weight_decay: float = WEIGHT_DECAY,
        epochs: int = EPOCHS,
        patience: int = 5,
        save_path: str = None,
    ):
        """
        Initialize the Trainer.

        Args:
            model: The HC-PFE model instance.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            device: Computation device (CPU or CUDA).
            max_lr: Maximum learning rate for OneCycleLR.
            weight_decay: Weight decay for AdamW.
            epochs: Total number of training epochs.
            patience: Patience for early stopping.
            save_path: Path to save the best model checkpoint.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.patience = patience
        self.save_path = (
            save_path if save_path else os.path.join(WORKING_DIR, "best_model.pth")
        )

        # Optimizer: AdamW
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=max_lr, weight_decay=weight_decay
        )

        # Scheduler: OneCycleLR
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
        )

        # Loss Function: BCEWithLogitsLoss
        self.criterion = nn.BCEWithLogitsLoss()

        # Metric Tracking
        self.best_auc = 0.0

    def train_epoch(self) -> float:
        """
        Runs one epoch of training.
        Returns:
            avg_loss (float): Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0

        for cat_x, cont_x, targets in self.train_loader:
            cat_x = cat_x.to(self.device)
            cont_x = cont_x.to(self.device)
            targets = targets.to(self.device).unsqueeze(1)

            self.optimizer.zero_grad()
            outputs = self.model(cat_x, cont_x)

            # Sum of independent BCE losses for all 5 streams
            loss = 0
            for out in outputs:
                loss += self.criterion(out, targets)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self) -> float:
        """
        Runs validation on the validation set.
        Returns:
            val_auc (float): Area Under the ROC Curve.
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for cat_x, cont_x, targets in self.val_loader:
                cat_x = cat_x.to(self.device)
                cont_x = cont_x.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(cat_x, cont_x)

                # Average probabilities from all 5 streams
                probs = [torch.sigmoid(out) for out in outputs]
                avg_prob = torch.mean(torch.stack(probs, dim=0), dim=0)

                all_preds.append(avg_prob.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        all_preds = np.concatenate(all_preds).flatten()
        all_targets = np.concatenate(all_targets).flatten()

        return roc_auc_score(all_targets, all_preds)

    def fit(self) -> str:
        """
        Executes the full training process with early stopping.
        Returns:
            best_model_path (str): Path to the saved best model.
        """
        print(f"Starting training on {self.device}...")
        patience_counter = 0

        for epoch in range(self.epochs):
            train_loss = self.train_epoch()
            val_auc = self.validate()

            # Print full precision metrics
            print(
                f"Epoch {epoch+1}/{self.epochs} | Train Loss: {train_loss:.10f} | Val AUC: {val_auc:.15f}"
            )

            if val_auc > self.best_auc:
                self.best_auc = val_auc
                save_checkpoint(self.model, self.save_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print("Early stopping triggered.")
                    break

        print(f"Best Val AUC: {self.best_auc:.15f}")
        return self.save_path
