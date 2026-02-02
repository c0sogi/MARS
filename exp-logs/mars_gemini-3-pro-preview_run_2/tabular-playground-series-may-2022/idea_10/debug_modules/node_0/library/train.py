import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from library.config import Config
from library.utils import seed_everything, compute_auc
from library.data_loader import get_dataloaders
from library.model import TransformerResFunnel


class Trainer:
    """
    Manages the training and validation of the Transformer-ResFunnel model.
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

    def train_epoch(self, train_loader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0

        for batch in train_loader:
            x_cont = batch["cont"].to(self.device)
            x_cat = batch["cat"].to(self.device)
            y = batch["target"].to(self.device).unsqueeze(1)

            self.optimizer.zero_grad()
            logits = self.model(x_cont, x_cat)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader):
        """
        Evaluates the model on the validation set.
        Returns average loss and AUC.
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                x_cont = batch["cont"].to(self.device)
                x_cat = batch["cat"].to(self.device)
                y = batch["target"].to(self.device).unsqueeze(1)

                logits = self.model(x_cont, x_cat)
                loss = self.criterion(logits, y)
                total_loss += loss.item()

                probs = torch.sigmoid(logits)
                all_preds.append(probs.cpu())
                all_targets.append(y.cpu())

        avg_loss = total_loss / len(val_loader)

        # Concatenate all batches
        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_targets).numpy()

        auc = compute_auc(y_true, y_pred)

        return avg_loss, auc


def run_training(debug=False, epochs=Config.EPOCHS):
    """
    Main execution function for training and submission generation.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running training on device: {device}")

    # 2. Data Loading
    # get_dataloaders handles caching internally
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=debug, load_cached_data=True
    )

    # 3. Model Initialization
    model = TransformerResFunnel().to(device)
    trainer = Trainer(model, device)

    # 4. Training Loop
    best_auc = 0.0
    patience_counter = 0

    print("Starting training loop...")
    for epoch in range(epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_loss, val_auc = trainer.validate(val_loader)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Early Stopping & Saving
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Training complete. Best Validation AUC: {best_auc}")

    # 5. Inference & Submission
    print("Generating submission...")

    # Load best model
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        print(
            "Warning: No model file found. Using current model state (likely untrained or failed)."
        )

    model.eval()
    test_preds = []

    with torch.no_grad():
        for batch in test_loader:
            x_cont = batch["cont"].to(device)
            x_cat = batch["cat"].to(device)

            logits = model(x_cont, x_cat)
            probs = torch.sigmoid(logits)
            test_preds.append(probs.cpu().numpy())

    # Flatten predictions
    test_preds = np.concatenate(test_preds).flatten()

    # Load test metadata to ensure correct ID alignment
    test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

    # Handle debug mode slicing
    if len(test_preds) != len(test_meta):
        if debug:
            print(
                f"Debug mode: Truncating metadata from {len(test_meta)} to {len(test_preds)} rows."
            )
            test_meta = test_meta.iloc[: len(test_preds)]
        else:
            print(
                f"Error: Prediction count {len(test_preds)} does not match metadata count {len(test_meta)}."
            )

    submission = pd.DataFrame({"id": test_meta["id"], "target": test_preds})

    # Save submission
    os.makedirs(os.path.dirname(Config.SUBMISSION_SAVE_PATH), exist_ok=True)
    submission.to_csv(Config.SUBMISSION_SAVE_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_SAVE_PATH}")
