import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import log_loss, accuracy_score
import copy
from library.config import Config


class ResNetBlock(nn.Module):
    """
    A Residual Network Block for Tabular Data.
    Structure: Input -> [Linear->BN->ReLU->Dropout] x2 -> Add Input -> ReLU
    Handles dimension changes via a projection layer in the shortcut.
    """

    def __init__(
        self, in_features, out_features, dropout_rate=0.0, use_batch_norm=True
    ):
        super(ResNetBlock, self).__init__()

        self.main_path = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features) if use_batch_norm else nn.Identity(),
            nn.Dropout(dropout_rate),
        )

        # Skip connection: Projection if dimensions change, else Identity
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features) if use_batch_norm else nn.Identity(),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.main_path(x)
        res = self.shortcut(x)
        # Standard ResNet: Addition before final ReLU
        return self.relu(out + res)


class ResNetMLP(nn.Module):
    """
    ResNet-MLP Architecture.
    Constructs a sequence of ResNetBlocks followed by a classification head.
    """

    def __init__(
        self,
        input_dim,
        num_classes,
        hidden_layers=None,
        dropout=0.0,
        use_batch_norm=True,
    ):
        super(ResNetMLP, self).__init__()

        if hidden_layers is None:
            hidden_layers = [512, 256, 128]

        layers = []
        current_dim = input_dim

        # Build Residual Blocks based on config
        for h_dim in hidden_layers:
            layers.append(
                ResNetBlock(
                    in_features=current_dim,
                    out_features=h_dim,
                    dropout_rate=dropout,
                    use_batch_norm=use_batch_norm,
                )
            )
            current_dim = h_dim

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(current_dim, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_nn_fold(train_loader, val_loader, input_dim, num_classes, params=None):
    """
    Trains the ResNetMLP model for a single fold.

    Args:
        train_loader (DataLoader): Loader for training data.
        val_loader (DataLoader): Loader for validation data.
        input_dim (int): Number of input features.
        num_classes (int): Number of target classes.
        params (dict, optional): Hyperparameter overrides.

    Returns:
        nn.Module: The trained model with the best validation state.
    """
    # Load default parameters and update with overrides
    nn_params = Config.NN_PARAMS.copy()
    if params:
        nn_params.update(params)

    device = torch.device(Config.DEVICE)

    # Initialize Model
    model = ResNetMLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_layers=nn_params["hidden_layers"],
        dropout=nn_params["dropout"],
        use_batch_norm=nn_params["use_batch_norm"],
    )
    model.to(device)

    # Optimizer (AdamW)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=nn_params["learning_rate"],
        weight_decay=nn_params["weight_decay"],
    )

    # Scheduler (Cosine Annealing)
    # Note: T_max is set to total epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=nn_params["epochs"], eta_min=1e-6
    )

    criterion = nn.CrossEntropyLoss()

    # Tracking variables
    best_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    print(f"Starting training on {device} for {nn_params['epochs']} epochs...")

    for epoch in range(nn_params["epochs"]):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        total_samples = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)
            total_samples += data.size(0)

        train_loss /= total_samples

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        total_val_samples = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item() * data.size(0)
                total_val_samples += data.size(0)

                # Collect predictions for metrics
                probs = torch.softmax(output, dim=1)
                all_preds.append(probs.cpu().numpy())
                all_targets.append(target.cpu().numpy())

        val_loss /= total_val_samples

        # --- Metrics Calculation ---
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        # Accuracy
        pred_labels = np.argmax(all_preds, axis=1)
        acc = accuracy_score(all_targets, pred_labels)

        # LogLoss
        ll = log_loss(all_targets, all_preds, labels=list(range(num_classes)))

        # Update Scheduler
        scheduler.step()

        # Print metrics with full precision
        print(
            f"Epoch {epoch + 1}/{nn_params['epochs']} - "
            f"Train Loss: {train_loss} - "
            f"Val Loss: {val_loss} - "
            f"Val Accuracy: {acc} - "
            f"Val LogLoss: {ll}"
        )

        # --- Early Stopping ---
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= nn_params["early_stopping_patience"]:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # Restore best model
    if best_model_state is not None:
        print(f"Restoring best model with Val Loss: {best_loss}")
        model.load_state_dict(best_model_state)

    return model


def predict_nn(model, test_loader):
    """
    Generates predictions using the trained Neural Network.

    Args:
        model (nn.Module): Trained model.
        test_loader (DataLoader): Loader for test data.

    Returns:
        np.ndarray: Predicted probabilities (N_samples, N_classes).
    """
    device = torch.device(Config.DEVICE)
    model.to(device)
    model.eval()

    all_preds = []

    with torch.no_grad():
        for data in test_loader:
            # Handle cases where loader might return (data, target) or just data
            if isinstance(data, (list, tuple)):
                data = data[0]

            data = data.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            all_preds.append(probs.cpu().numpy())

    return np.concatenate(all_preds, axis=0)
