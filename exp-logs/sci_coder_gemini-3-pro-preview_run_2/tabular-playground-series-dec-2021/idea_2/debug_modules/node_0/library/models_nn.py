import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import library.config as config
import os

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TabularDataset(Dataset):
    """
    PyTorch Dataset for tabular data.
    """

    def __init__(self, X, y=None):
        # Convert DataFrame to float32 tensor
        if isinstance(X, pd.DataFrame):
            self.X = torch.tensor(X.values, dtype=torch.float32)
        else:
            self.X = torch.tensor(X, dtype=torch.float32)

        if y is not None:
            self.y = torch.tensor(y, dtype=torch.long)
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


class ResBlock(nn.Module):
    """
    A single residual block for the TabularResNet.
    Structure: Linear -> BN -> ReLU -> Dropout.
    Adds a skip connection if input_dim == output_dim.
    """

    def __init__(self, in_features, out_features, dropout_rate):
        super(ResBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        # Determine if we can apply a residual connection
        self.has_residual = in_features == out_features

    def forward(self, x):
        out = self.linear(x)
        out = self.bn(out)
        out = self.activation(out)
        out = self.dropout(out)

        if self.has_residual:
            return x + out
        else:
            return out


class TabularResNet(nn.Module):
    """
    MLP with Residual connections.
    """

    def __init__(self, input_dim, output_dim, hidden_layers, dropout_rate):
        super(TabularResNet, self).__init__()

        layers = []
        current_dim = input_dim

        for h_dim in hidden_layers:
            layers.append(ResBlock(current_dim, h_dim, dropout_rate))
            current_dim = h_dim

        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(current_dim, output_dim)

    def forward(self, x):
        x = self.layers(x)
        x = self.output_layer(x)
        return x


class NNWrapper:
    """
    Wrapper class to handle training and inference of the Neural Network.
    """

    def __init__(self, params=None):
        self.params = params if params is not None else config.NN_PARAMS.copy()
        self.model = None
        self.device = DEVICE
        self._set_seed(self.params.get("seed", 42))

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the neural network with early stopping.
        """
        # Update input_dim based on data
        input_dim = X_train.shape[1]
        self.params["input_dim"] = input_dim

        # Initialize Model
        self.model = TabularResNet(
            input_dim=self.params["input_dim"],
            output_dim=self.params["output_dim"],
            hidden_layers=self.params["hidden_layers"],
            dropout_rate=self.params["dropout"],
        ).to(self.device)

        # Optimization
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.params["learning_rate"],
            weight_decay=self.params["weight_decay"],
        )

        criterion = nn.CrossEntropyLoss()

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self.params["scheduler_factor"],
            patience=self.params["scheduler_patience"],
            verbose=False,
        )

        # Data Loaders
        train_dataset = TabularDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.params["batch_size"],
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = TabularDataset(X_val, y_val)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.params["batch_size"] * 2,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
            )

        # Training Loop
        best_acc = 0.0
        patience_counter = 0
        best_model_state = None

        epochs = self.params["epochs"]

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_X.size(0)

            train_loss /= len(train_dataset)

            # Validation
            if val_loader:
                val_acc, val_loss = self._evaluate(val_loader, criterion)
                scheduler.step(val_acc)

                # Checkpointing
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_model_state = self.model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.params["early_stopping_patience"]:
                    # print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                # If no validation set, just save the last state
                best_model_state = self.model.state_dict()

        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        if X_val is not None and y_val is not None:
            print(f"Final Validation Accuracy: {best_acc}")

    def _evaluate(self, loader, criterion):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                running_loss += loss.item() * batch_X.size(0)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())

        total_loss = running_loss / len(loader.dataset)
        accuracy = accuracy_score(all_targets, all_preds)
        return accuracy, total_loss

    def predict_proba(self, X):
        """
        Predict class probabilities.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        dataset = TabularDataset(X)
        loader = DataLoader(
            dataset,
            batch_size=self.params["batch_size"] * 2,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        self.model.eval()
        all_probs = []

        with torch.no_grad():
            for batch_X in loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())

        return np.vstack(all_probs)
