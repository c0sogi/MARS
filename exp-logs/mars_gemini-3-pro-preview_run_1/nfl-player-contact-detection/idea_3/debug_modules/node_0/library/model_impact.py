import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import matthews_corrcoef, log_loss
from library.config import Config
from library.utils import seed_everything, print_metric


class ImpactDataset(Dataset):
    """
    PyTorch Dataset for Stream B (Impact Model).
    Handles 3D tensor inputs (Batch, Channels, Time) and binary targets.
    """

    def __init__(self, X, y=None):
        """
        Args:
            X (np.ndarray): Input features of shape (N, Channels, Time).
            y (np.ndarray, optional): Target labels of shape (N,).
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


class ResidualBlock1D(nn.Module):
    """
    1D Residual Block with Batch Normalization and Dropout.
    Structure: Conv -> BN -> ReLU -> Dropout -> Conv -> BN -> (+ Input) -> ReLU
    """

    def __init__(self, channels, kernel_size, dropout):
        super(ResidualBlock1D, self).__init__()
        # Padding to maintain sequence length
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class Impact1DResNet(nn.Module):
    """
    1D ResNet for detecting ground impacts from kinematic sequences.
    """

    def __init__(self, config=Config):
        super(Impact1DResNet, self).__init__()
        params = config.CNN_PARAMS

        input_channels = params["input_channels"]
        hidden_dim = params["hidden_dim"]
        kernel_size = params["kernel_size"]
        dropout = params["dropout"]

        # Initial Projection
        padding = (kernel_size - 1) // 2
        self.conv_in = nn.Conv1d(
            input_channels, hidden_dim, kernel_size, padding=padding, bias=False
        )
        self.bn_in = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()

        # Residual Blocks
        # Stacking two blocks for sufficient depth to capture temporal impulse patterns
        self.layer1 = ResidualBlock1D(hidden_dim, kernel_size, dropout)
        self.layer2 = ResidualBlock1D(hidden_dim, kernel_size, dropout)

        # Classification Head
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.head_drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim // 2, 1)  # Output logits

    def forward(self, x):
        # x shape: (Batch, Channels, Time)
        x = self.conv_in(x)
        x = self.bn_in(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.pool(x)
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.head_drop(x)
        x = self.fc2(x)
        return x


class ImpactTrainer:
    """
    Handles training, validation, and inference for the Impact1DResNet model.
    """

    def __init__(self, config=Config):
        self.config = config
        seed_everything(self.config.SEED)

        self.device = torch.device(self.config.CNN_TRAIN_PARAMS["device"])
        self.model_path = os.path.join(self.config.WORKING_DIR, "cnn_impact_model.pth")

        self.model = Impact1DResNet(config).to(self.device)

        # Training components
        self.criterion = nn.BCEWithLogitsLoss()

        lr = self.config.CNN_TRAIN_PARAMS["learning_rate"]
        weight_decay = self.config.CNN_TRAIN_PARAMS["weight_decay"]
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

    def train(self, X_train, y_train, X_val, y_val):
        """
        Trains the model with Early Stopping.

        Args:
            X_train, y_train: Training features and targets.
            X_val, y_val: Validation features and targets.
        """
        print(f"Initializing ImpactTrainer on device: {self.device}")

        # Create DataLoaders
        train_dataset = ImpactDataset(X_train, y_train)
        val_dataset = ImpactDataset(X_val, y_val)

        batch_size = self.config.CNN_TRAIN_PARAMS["batch_size"]
        num_workers = self.config.CNN_TRAIN_PARAMS["num_workers"]

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        epochs = self.config.CNN_TRAIN_PARAMS["epochs"]
        scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

        # Early Stopping variables
        best_val_loss = float("inf")
        patience = 5
        patience_counter = 0

        print(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(
                    self.device
                ).unsqueeze(1)

                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * X_batch.size(0)

            avg_train_loss = train_loss / len(train_dataset)

            # Validation
            val_loss, val_mcc = self._evaluate(val_loader)

            # Scheduler step
            scheduler.step()

            # Checkpoint & Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.model_path)
                # Verbose print for best model
                print(
                    f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f} - Val MCC: {val_mcc:.4f} [Saved]"
                )
            else:
                patience_counter += 1
                if patience_counter % 2 == 0:  # Print occasionally if not improving
                    print(
                        f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f} - Val MCC: {val_mcc:.4f}"
                    )

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        print(f"Training complete. Best Val Loss: {best_val_loss:.6f}")
        # Print final metrics with full precision as requested
        print_metric("Best Validation LogLoss", best_val_loss)

    def _evaluate(self, dataloader):
        """
        Internal evaluation loop.
        Returns: avg_loss, mcc
        """
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device).unsqueeze(1)

                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)

                running_loss += loss.item() * X_batch.size(0)

                probs = torch.sigmoid(outputs).cpu().numpy()
                targets = y_batch.cpu().numpy()

                all_preds.append(probs)
                all_targets.append(targets)

        avg_loss = running_loss / len(dataloader.dataset)

        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)

        # Calculate MCC with 0.5 threshold
        preds_binary = (all_preds > 0.5).astype(int)
        mcc = matthews_corrcoef(all_targets, preds_binary)

        return avg_loss, mcc

    def predict(self, X):
        """
        Generates predictions for input features.
        Loads the best model from disk if available.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Probabilities of contact.
        """
        # Load best model
        if os.path.exists(self.model_path):
            print(f"Loading Impact Model from {self.model_path}...")
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )
        else:
            print(
                "Warning: No saved model found. Using current weights (random or last epoch)."
            )

        dataset = ImpactDataset(X)
        batch_size = self.config.CNN_TRAIN_PARAMS["batch_size"]
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.CNN_TRAIN_PARAMS["num_workers"],
        )

        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for X_batch in loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_preds.append(probs)

        return np.vstack(all_preds).flatten()
