import os
import math
import torch
import numpy as np
import pandas as pd
from library.config import CONFIG


class Engine:
    """
    Engine class to handle training, validation, and inference for the PA-DSDS model.
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        scheduler=None,
        patience=CONFIG["patience"],
        save_dir="./working/idea_6",
    ):
        """
        Args:
            model (nn.Module): The PyTorch model to train.
            optimizer (torch.optim.Optimizer): The optimizer.
            criterion (nn.Module): The loss function.
            device (torch.device): Device to run on (cpu or cuda).
            scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler.
            patience (int): Early stopping patience.
            save_dir (str): Directory to save the best model checkpoint.
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.patience = patience
        self.save_dir = save_dir

        # Ensure save directory exists
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_model_path = os.path.join(self.save_dir, "best_model.pt")

        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def train_one_epoch(self, dataloader):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        running_loss = 0.0
        total_samples = 0

        for batch in dataloader:
            # Unpack batch: atomic_feats, batch_indices, global_feats, targets, ids
            atomic_feats, batch_indices, global_feats, targets, _ = batch

            # Move to device
            atomic_feats = atomic_feats.to(self.device)
            batch_indices = batch_indices.to(self.device)
            global_feats = global_feats.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(atomic_feats, batch_indices, global_feats)

            # Compute loss
            loss = self.criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            # Accumulate loss (weighted by batch size)
            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

        avg_loss = running_loss / total_samples
        return avg_loss

    def validate(self, dataloader):
        """
        Evaluates the model on the validation set.
        Returns average loss and RMSLE.
        """
        self.model.eval()
        running_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                atomic_feats, batch_indices, global_feats, targets, _ = batch

                atomic_feats = atomic_feats.to(self.device)
                batch_indices = batch_indices.to(self.device)
                global_feats = global_feats.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(atomic_feats, batch_indices, global_feats)
                loss = self.criterion(outputs, targets)

                batch_size = targets.size(0)
                running_loss += loss.item() * batch_size
                total_samples += batch_size

        avg_loss = running_loss / total_samples

        # Since targets are log(1+x), MSE on log scale corresponds to MSLE on original scale.
        # RMSLE is simply the square root of the MSE loss on log-transformed targets.
        rmsle = math.sqrt(avg_loss)

        return avg_loss, rmsle

    def fit(self, train_loader, val_loader, epochs=CONFIG["epochs"]):
        """
        Runs the full training loop with early stopping.
        """
        print(f"Starting training for {epochs} epochs on {self.device}...")

        for epoch in range(epochs):
            train_loss = self.train_one_epoch(train_loader)
            val_loss, val_rmsle = self.validate(val_loader)

            # Print metrics with full precision
            print(
                f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val RMSLE: {val_rmsle}"
            )

            # Step scheduler
            if self.scheduler:
                self.scheduler.step(val_loss)

            # Early Stopping Check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        # Load best model weights
        if os.path.exists(self.best_model_path):
            self.model.load_state_dict(
                torch.load(self.best_model_path, map_location=self.device)
            )
            print("Best model loaded.")
        else:
            print("Warning: No best model saved.")

    def predict(self, dataloader):
        """
        Generates predictions for the given dataloader.
        Applies inverse transformation (expm1) to the outputs.
        """
        self.model.eval()
        all_preds = []
        all_ids = []

        with torch.no_grad():
            for batch in dataloader:
                atomic_feats, batch_indices, global_feats, _, ids = batch

                atomic_feats = atomic_feats.to(self.device)
                batch_indices = batch_indices.to(self.device)
                global_feats = global_feats.to(self.device)

                outputs = self.model(atomic_feats, batch_indices, global_feats)

                # Inverse transform: log(1+x) -> exp(y) - 1
                preds = torch.expm1(outputs)

                all_preds.append(preds.cpu().numpy())
                all_ids.extend(ids)

        if len(all_preds) > 0:
            return np.concatenate(all_preds, axis=0), all_ids
        else:
            return np.array([]), []

    def generate_submission(
        self, test_loader, output_path="./submission/submission.csv"
    ):
        """
        Generates predictions for the test set and saves them to a CSV file.
        """
        print("Generating submission...")
        preds, ids = self.predict(test_loader)

        if len(preds) == 0:
            print("No predictions generated.")
            return

        # Create DataFrame
        df = pd.DataFrame(
            preds, columns=["formation_energy_ev_natom", "bandgap_energy_ev"]
        )
        df.insert(0, "id", ids)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
