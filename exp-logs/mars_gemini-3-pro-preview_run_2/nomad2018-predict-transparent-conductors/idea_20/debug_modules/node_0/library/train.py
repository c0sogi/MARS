import torch
import torch.nn as nn
import pandas as pd
import os
import numpy as np

# Import from provided libraries
from library.model import AICGN
from library.data import get_dataloaders
from library.utils import set_seed, compute_rmsle, StandardScaler


class Trainer:
    """
    Manages the training and evaluation loops for the AICGN model.
    """

    def __init__(
        self,
        model,
        device,
        learning_rate=1e-3,
        weight_decay=1e-4,
        patience=15,
        checkpoint_dir="./working/checkpoints",
    ):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()
        self.scaler = StandardScaler(device)
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_model_path = os.path.join(self.checkpoint_dir, "best_model.pth")

    def fit_scaler(self, train_loader):
        """
        Fits the target scaler on the training data.
        """
        all_targets = []
        for data in train_loader:
            all_targets.append(data.y)
        all_targets = torch.cat(all_targets, dim=0)
        self.scaler.fit(all_targets)

    def train_epoch(self, train_loader):
        """
        Performs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0
        num_graphs = 0

        for data in train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()

            pred_form, pred_band = self.model(data)
            pred = torch.cat([pred_form, pred_band], dim=1)

            # Transform targets
            target_norm = self.scaler.transform(data.y)

            loss = self.criterion(pred, target_norm)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * data.num_graphs
            num_graphs += data.num_graphs

        return total_loss / num_graphs

    def evaluate(self, val_loader):
        """
        Evaluates the model on the validation set.
        Returns average loss and RMSLE.
        """
        self.model.eval()
        total_loss = 0.0
        num_graphs = 0

        val_preds = []
        val_targets = []

        with torch.no_grad():
            for data in val_loader:
                data = data.to(self.device)
                pred_form, pred_band = self.model(data)
                pred_norm = torch.cat([pred_form, pred_band], dim=1)

                # Loss on normalized data
                target_norm = self.scaler.transform(data.y)
                loss = self.criterion(pred_norm, target_norm)
                total_loss += loss.item() * data.num_graphs
                num_graphs += data.num_graphs

                # Inverse transform for RMSLE calculation
                pred_orig = self.scaler.inverse_transform(pred_norm)
                val_preds.append(pred_orig)
                val_targets.append(data.y)

        avg_loss = total_loss / num_graphs

        val_preds = torch.cat(val_preds, dim=0)
        val_targets = torch.cat(val_targets, dim=0)

        # Calculate RMSLE on original scale
        val_rmsle = compute_rmsle(val_preds, val_targets)

        return avg_loss, val_rmsle

    def fit(self, train_loader, val_loader, epochs=100):
        """
        Main training loop with early stopping.
        """
        # Fit scaler first
        self.fit_scaler(train_loader)

        best_val_loss = float("inf")
        counter = 0

        print("Starting training...")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_rmsle = self.evaluate(val_loader)

            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val RMSLE: {val_rmsle}"
            )

            # Checkpoint and Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.best_model_path)
                counter = 0
            else:
                counter += 1
                if counter >= self.patience:
                    print("Early stopping triggered.")
                    break

        # Load best model
        self.model.load_state_dict(torch.load(self.best_model_path))
        print(f"Best model loaded from {self.best_model_path}")

    def predict(self, loader):
        """
        Generates predictions for a given loader.
        Returns IDs and predictions (original scale).
        """
        self.model.eval()
        ids = []
        preds = []

        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                p_form, p_band = self.model(data)
                pred_norm = torch.cat([p_form, p_band], dim=1)
                pred_orig = self.scaler.inverse_transform(pred_norm)

                # Ensure non-negative
                pred_orig = torch.clamp(pred_orig, min=0.0)

                ids.extend(data.id.cpu().numpy())
                preds.append(pred_orig.cpu().numpy())

        return np.array(ids), np.concatenate(preds, axis=0)


def train_and_evaluate(
    batch_size=48,
    hidden_dim=128,
    num_layers=4,
    dropout=0.1,
    epochs=150,
    lr=1e-3,
    weight_decay=1e-4,
    seed=42,
    sample_size=None,
    load_cached_data=True,
):
    """
    Orchestrates the training process.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Loading
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=2,
        load_cached_data=load_cached_data,
        sample_size=sample_size,
    )

    # Model Initialization
    model = AICGN(
        node_input_dim=100,
        edge_input_dim=60,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    # Trainer Initialization
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=lr,
        weight_decay=weight_decay,
        patience=15,
    )

    # Training
    trainer.fit(train_loader, val_loader, epochs=epochs)

    return trainer, test_loader


def generate_submission_file(
    trainer, test_loader, output_path="./submission/submission.csv"
):
    """
    Generates the submission file using the trained model.
    """
    ids, preds = trainer.predict(test_loader)

    df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": preds[:, 0],
            "bandgap_energy_ev": preds[:, 1],
        }
    )

    # Sort by ID
    df = df.sort_values("id")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
