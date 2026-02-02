import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import compute_metric, save_checkpoint, load_checkpoint


class MaskedL1Loss(nn.Module):
    """
    Computes L1 Loss only for time steps where u_out == 0 (Inspiratory Phase).
    """

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction="none")

    def forward(self, pred, target, u_out):
        """
        Args:
            pred: (Batch, Seq_Len, 1) or (Batch, Seq_Len)
            target: (Batch, Seq_Len)
            u_out: (Batch, Seq_Len) - 0 for inspiratory, 1 for expiratory
        """
        if pred.dim() == 3:
            pred = pred.squeeze(-1)

        loss = self.l1(pred, target)

        # Mask out expiratory phase (u_out == 1)
        # u_out is float, so we use 1.0 - u_out to get 1.0 for inspiratory
        mask = 1.0 - u_out

        masked_loss = loss * mask

        # Sum of losses / Sum of mask (number of valid items)
        # Add epsilon to denominator to prevent division by zero
        return masked_loss.sum() / (mask.sum() + 1e-8)


def train_one_epoch(model, dataloader, optimizer, scheduler, device, criterion):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        x, y, u_out = batch
        x = x.to(device)
        y = y.to(device)
        u_out = u_out.to(device)

        optimizer.zero_grad()

        preds = model(x)
        loss = criterion(preds, y, u_out)

        loss.backward()

        # Gradient Clipping to prevent exploding gradients in LSTM
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model, dataloader, device, criterion):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            x, y, u_out = batch
            x = x.to(device)
            y = y.to(device)
            u_out = u_out.to(device)

            preds = model(x)
            loss = criterion(preds, y, u_out)

            # compute_metric handles the masking internally for the metric
            mae = compute_metric(preds, y, u_out)

            total_loss += loss.item()
            total_mae += mae
            num_batches += 1

    return total_loss / num_batches, total_mae / num_batches


class Engine:
    """
    Main class to handle training, evaluation, and inference.
    """

    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.criterion = MaskedL1Loss()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        self.scheduler = None

    def fit(self, train_loader, val_loader):
        """
        Runs the training loop with Early Stopping.
        """
        epochs = Config.EPOCHS
        # Setup OneCycleLR scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=Config.LEARNING_RATE,
            steps_per_epoch=len(train_loader),
            epochs=epochs,
            pct_start=Config.PCT_START,
            div_factor=Config.DIV_FACTOR,
            final_div_factor=Config.FINAL_DIV_FACTOR,
        )

        best_mae = float("inf")
        patience = 7  # Early stopping patience
        patience_counter = 0

        print(f"Starting training for {epochs} epochs on {self.device}...")

        for epoch in range(epochs):
            train_loss = train_one_epoch(
                self.model,
                train_loader,
                self.optimizer,
                self.scheduler,
                self.device,
                self.criterion,
            )

            val_loss, val_mae = evaluate(
                self.model, val_loader, self.device, self.criterion
            )

            # Print full precision metrics
            print(
                f"Epoch {epoch+1} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val MAE: {val_mae}"
            )

            if val_mae < best_mae:
                best_mae = val_mae
                patience_counter = 0
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": self.model.state_dict(),
                        "best_loss": best_mae,
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict(),
                    },
                    Config.MODEL_PATH,
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(
                        f"Early stopping triggered at epoch {epoch+1}. Best MAE: {best_mae}"
                    )
                    break

        print(f"Training finished. Best Validation MAE: {best_mae}")

        # Load best model weights for subsequent prediction
        print(f"Loading best model from {Config.MODEL_PATH}...")
        load_checkpoint(Config.MODEL_PATH, self.model, device=self.device)

    def predict(self, test_loader):
        """
        Generates predictions for the test set.
        """
        self.model.eval()
        all_ids = []
        all_preds = []

        with torch.no_grad():
            for batch in test_loader:
                x, u_out, ids = batch
                x = x.to(self.device)

                preds = self.model(x)

                # Flatten predictions and IDs
                # preds: (B, 80, 1) -> (B*80)
                preds_flat = preds.view(-1).cpu().numpy()
                ids_flat = ids.view(-1).cpu().numpy()

                all_ids.append(ids_flat)
                all_preds.append(preds_flat)

        return np.concatenate(all_ids), np.concatenate(all_preds)

    def generate_submission(self, test_loader):
        """
        Generates and saves the submission CSV.
        """
        print("Generating submission...")

        ids, preds = self.predict(test_loader)

        df = pd.DataFrame({"id": ids, "pressure": preds})

        # Ensure sorting by ID
        df = df.sort_values("id")

        df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
