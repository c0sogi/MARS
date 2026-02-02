import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import set_seed
from library.mlp_model import DecoupledGatedMLP, PizzaDataset


class MLPTrainer:
    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        self.model = DecoupledGatedMLP().to(self.device)
        self.model_path = Config.MODEL_MLP_PATH

        # Training Config
        self.batch_size = Config.MLP_BATCH_SIZE
        self.epochs = Config.MLP_EPOCHS
        self.patience = Config.MLP_PATIENCE
        self.learning_rate = Config.MLP_LEARNING_RATE
        self.weight_decay = Config.MLP_WEIGHT_DECAY

        set_seed(Config.RANDOM_SEED)

    def train(self, train_data, val_data):
        """
        Trains the MLP model with Early Stopping.

        Args:
            train_data: tuple (feature_dict, labels)
            val_data: tuple (feature_dict, labels)
        """
        # Prepare Datasets
        train_dataset = PizzaDataset(train_data[0], train_data[1])
        val_dataset = PizzaDataset(val_data[0], val_data[1])

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        # Optimizer & Loss
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss()

        # Early Stopping State
        best_auc = 0.0
        patience_counter = 0

        print(f"Starting MLP training on {self.device}...")

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0

            for batch_inputs, batch_labels in train_loader:
                # Move to device
                batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_inputs)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_labels.size(0)

            avg_train_loss = train_loss / len(train_dataset)

            # Validation
            val_auc, val_loss = self._evaluate(val_loader, criterion)

            print(
                f"Epoch {epoch+1}/{self.epochs} | "
                f"Train Loss: {avg_train_loss} | "
                f"Val Loss: {val_loss} | "
                f"Val AUC: {val_auc}"
            )

            # Early Stopping Check
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                self._save_model()
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break

        # Load best model for final state
        self._load_model()
        print(f"Training complete. Best Val AUC: {best_auc}")

    def predict(self, feature_dict):
        """
        Generates predictions for the given features.

        Args:
            feature_dict: Dictionary of features
        Returns:
            np.ndarray: Probabilities
        """
        if not os.path.exists(self.model_path):
            self._load_model()  # Try loading, might fail if not trained

        dataset = PizzaDataset(feature_dict)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch_inputs in loader:
                batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}
                logits = self.model(batch_inputs)
                probs = torch.sigmoid(logits)
                all_preds.append(probs.cpu().numpy())

        return np.concatenate(all_preds)

    def _evaluate(self, loader, criterion):
        self.model.eval()
        total_loss = 0.0
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch_inputs, batch_labels in loader:
                batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}
                batch_labels = batch_labels.to(self.device)

                logits = self.model(batch_inputs)
                loss = criterion(logits, batch_labels)

                total_loss += loss.item() * batch_labels.size(0)
                probs = torch.sigmoid(logits)

                all_labels.append(batch_labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        avg_loss = total_loss / len(loader.dataset)
        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)

        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.0

        return auc, avg_loss

    def _save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)

    def _load_model(self):
        if os.path.exists(self.model_path):
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )
        else:
            print(f"Warning: Model file not found at {self.model_path}")
