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


def train_classifier(
    model,
    train_loader,
    val_loader,
    epochs=Config.EPOCHS,
    lr=Config.LR,
    patience=Config.PATIENCE,
    device=Config.DEVICE,
):
    """
    Trains the Classifier (ResNet-MLP) with Early Stopping and Scheduler.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Scheduler: Reduce LR when validation loss stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.SCHEDULER_MIN_LR,
        verbose=True,
    )

    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=patience, mode="max")  # Maximize Accuracy

    print("Starting Classifier Training...")

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

        # Step Scheduler
        scheduler.step(val_loss)

        # Check Early Stopping
        early_stopping(val_acc, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Load best weights
    early_stopping.load_best_weights(model)
    return model
