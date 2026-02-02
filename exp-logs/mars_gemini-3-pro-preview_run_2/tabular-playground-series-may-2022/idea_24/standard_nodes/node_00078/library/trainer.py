import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from library.model import GatedStemHybridNet


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(
        self,
        model,
        device,
        learning_rate=1e-3,
        weight_decay=1e-2,
        step_size=10,
        gamma=0.1,
    ):
        """
        Args:
            model (nn.Module): The PyTorch model to train.
            device (torch.device): Device to run training on.
            learning_rate (float): Initial learning rate for AdamW.
            weight_decay (float): Weight decay for AdamW.
            step_size (int): Period of learning rate decay.
            gamma (float): Multiplicative factor of learning rate decay.
        """
        self.model = model.to(device)
        self.device = device

        # Optimizer: AdamW with high weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Scheduler: Aggressive Step Learning Rate Scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=step_size, gamma=gamma
        )

        # Loss: BCEWithLogitsLoss for numerical stability
        self.criterion = nn.BCEWithLogitsLoss()

        self.best_auc = -float("inf")

    def train_one_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0

        for batch_idx, (x_cont, x_seq, targets) in enumerate(train_loader):
            x_cont = x_cont.to(self.device)
            x_seq = x_seq.to(self.device)
            targets = targets.to(self.device).unsqueeze(
                1
            )  # Match output shape (Batch, 1)

            self.optimizer.zero_grad()

            logits = self.model(x_cont, x_seq)
            loss = self.criterion(logits, targets)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * x_cont.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        return epoch_loss

    def validate(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for x_cont, x_seq, targets in val_loader:
                x_cont = x_cont.to(self.device)
                x_seq = x_seq.to(self.device)
                targets = targets.to(self.device).unsqueeze(1)

                logits = self.model(x_cont, x_seq)
                loss = self.criterion(logits, targets)

                running_loss += loss.item() * x_cont.size(0)

                # Apply sigmoid for probabilities
                probs = torch.sigmoid(logits)

                all_targets.append(targets.cpu().numpy())
                all_preds.append(probs.cpu().numpy())

        val_loss = running_loss / len(val_loader.dataset)

        all_targets = np.concatenate(all_targets)
        all_preds = np.concatenate(all_preds)

        try:
            auc = roc_auc_score(all_targets, all_preds)
        except ValueError:
            auc = 0.5  # Fallback if only one class present in batch

        return val_loss, auc

    def fit(self, train_loader, val_loader, epochs=40, checkpoint_dir="./working"):
        """
        Main training loop.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")

        print(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            train_loss = self.train_one_epoch(train_loader)
            val_loss, val_auc = self.validate(val_loader)

            # Step the scheduler
            current_lr = self.scheduler.get_last_lr()[0]
            self.scheduler.step()

            # Print full precision metrics
            print(
                f"Epoch {epoch+1}/{epochs} | LR: {current_lr:.2e} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc}"
            )

            # Checkpoint based on AUC
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"New best AUC! Model saved to {checkpoint_path}")

        print(f"Training complete. Best Validation AUC: {self.best_auc}")

    def predict(self, test_loader, checkpoint_path="./working/best_model.pth"):
        """
        Loads the best model and generates predictions for the test set.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        print(f"Loading best model from {checkpoint_path}...")
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device)
        )
        self.model.eval()

        all_preds = []

        with torch.no_grad():
            for x_cont, x_seq in test_loader:
                x_cont = x_cont.to(self.device)
                x_seq = x_seq.to(self.device)

                logits = self.model(x_cont, x_seq)
                probs = torch.sigmoid(logits)

                all_preds.append(probs.cpu().numpy())

        return np.concatenate(all_preds).flatten()
