import copy
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import print_metric


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs. Saves the best model state.
    """

    def __init__(self, patience=Config.PATIENCE, mode="min", min_delta=0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state = None

    def __call__(self, metric, model):
        # Convert metric to score where higher is better
        if self.mode == "min":
            score = -metric
        else:
            score = metric

        if self.best_score is None:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0

    def load_best_weights(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def calculate_dae_loss(recon, target):
    """
    Calculates the composite loss for DAE: MSE for continuous, BCE for binary.
    """
    # Determine split index
    # Continuous cols: 10 original + 2 engineered = 12
    num_cont = len(Config.CONTINUOUS_COLS) + 2

    # Split tensors
    recon_cont = recon[:, :num_cont]
    recon_bin = recon[:, num_cont:]

    target_cont = target[:, :num_cont]
    target_bin = target[:, num_cont:]

    # Define criteria
    mse_crit = nn.MSELoss()
    bce_crit = nn.BCEWithLogitsLoss()

    # Calculate losses
    loss_mse = mse_crit(recon_cont, target_cont)
    loss_bce = bce_crit(recon_bin, target_bin)

    # Weighted sum
    total_loss = (Config.MSE_WEIGHT * loss_mse) + (Config.BCE_WEIGHT * loss_bce)
    return total_loss, loss_mse.item(), loss_bce.item()


def train_dae(
    model,
    train_loader,
    val_loader,
    epochs=Config.EPOCHS_PRETRAIN,
    lr=Config.LR_PRETRAIN,
    device=Config.DEVICE,
):
    """
    Trains the Denoising Autoencoder.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Starting DAE Pretraining...")

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            # Handle different dataset returns
            if isinstance(batch, (list, tuple)):
                if len(batch) == 2:
                    x_noisy, x_clean = batch
                else:
                    x_noisy = batch[0]
                    x_clean = batch[0]
            else:
                x_noisy = batch
                x_clean = batch

            x_noisy = x_noisy.to(device)
            x_clean = x_clean.to(device)

            optimizer.zero_grad()

            # Forward pass (returns recon, latent)
            recon, _ = model(x_noisy)

            loss, _, _ = calculate_dae_loss(recon, x_clean)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_noisy.size(0)

        train_loss /= len(train_loader.dataset)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        val_bce = 0.0

        with torch.no_grad():
            for batch in val_loader:
                # Determine input and target for validation
                # If batch is (x, y_label), use x for both recon input and target
                # If batch is (noisy, clean), use noisy input and clean target

                if isinstance(batch, (list, tuple)):
                    elem1, elem2 = batch
                    # Check if second element is target label (long/int)
                    if elem2.dtype == torch.long:
                        x_in = elem1
                        target = elem1
                    else:
                        x_in = elem1
                        target = elem2
                else:
                    x_in = batch
                    target = batch

                x_in = x_in.to(device)
                target = target.to(device)

                recon, _ = model(x_in)
                loss, mse, bce = calculate_dae_loss(recon, target)

                bs = x_in.size(0)
                val_loss += loss.item() * bs
                val_mse += mse * bs
                val_bce += bce * bs

        val_loss /= len(val_loader.dataset)
        val_mse /= len(val_loader.dataset)
        val_bce /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{epochs}")
        print_metric("DAE Train Loss", train_loss)
        print_metric("DAE Val Loss", val_loss)
        print_metric("DAE Val MSE", val_mse)
        print_metric("DAE Val BCE", val_bce)

    return model


def train_classifier(
    model,
    train_loader,
    val_loader,
    epochs=Config.EPOCHS_FINETUNE,
    lr=Config.LR_FINETUNE,
    patience=Config.PATIENCE,
    device=Config.DEVICE,
):
    """
    Trains the Classifier (ResNet-MLP) with Early Stopping.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=patience, mode="max")  # Maximize Accuracy

    print("Starting Classifier Fine-tuning...")

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss /= total
        train_acc = correct / total

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                logits = model(x)
                loss = criterion(logits, y)

                val_loss += loss.item() * x.size(0)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{epochs}")
        print_metric("Train Loss", train_loss)
        print_metric("Train Acc", train_acc)
        print_metric("Val Loss", val_loss)
        print_metric("Val Acc", val_acc)

        # Check Early Stopping
        early_stopping(val_acc, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Load best weights
    early_stopping.load_best_weights(model)
    return model
